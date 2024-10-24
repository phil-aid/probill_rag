from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb.utils.embedding_functions as embedding_functions
from tqdm import tqdm
import pandas as pd
import chromadb
from chromadb.config import Settings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from probill.probill.utils.logging_utils import log

class VectorStoreManager:
    def __init__(self, data_path, model_name="nomic-embed-text:latest", filter_key=None, batch_size=500, recreate=False):
        log("Initializing ...")
        self.model_name = model_name
        self.filter_key = filter_key
        self.batch_size = batch_size
        self.recreate = recreate
        self.client = chromadb.HttpClient(settings=Settings(allow_reset=True), host="10.0.40.49", port=8003)
        self.ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://10.0.40.49:11434/api/embeddings",
            model_name=model_name,
        )
        self.embedding = OllamaEmbeddings(model=self.model_name, base_url="http://10.0.40.49:11434")

        self._prepare_data(data_path)
        self.collection = self._get_or_create_vectorstore()

    def _prepare_data(self, data_path):
        log("Preparing data...")
        columns_to_include = ["icd10_code_id", "description"]
        
        # Use chunksize for memory efficiency
        chunks = pd.read_csv(data_path, usecols=columns_to_include, chunksize=100000)
        
        filtered_chunks = []
        for chunk in chunks:
            if self.filter_key:
                chunk = chunk[chunk['icd10_code_id'].str.startswith(self.filter_key)]
            filtered_chunks.append(chunk)
        
        self.df = pd.concat(filtered_chunks).drop_duplicates(subset=['icd10_code_id'], keep='last')
        log(f"Prepared {len(self.df)} records")

    def _get_or_create_vectorstore(self):
        collection_name = f"ICD_10_CODE_{self.filter_key}" if self.filter_key else "ICD_10_CODE"
        
        if self.recreate:
            log(f"Recreating vector store for {collection_name}...", log_level="INFO")
            self.client.delete_collection(collection_name)
            collection = self.client.create_collection(collection_name, embedding_function=self.ollama_ef)
            self._populate_vectorstore(collection)
        else:
            collection = self.client.get_or_create_collection(collection_name, embedding_function=self.ollama_ef)
            if collection.count() == 0:
                log(f"Vector store for {collection_name} is empty. Initializing...", log_level="INFO")
                self._populate_vectorstore(collection)
            else:
                log(f"Vector store for {collection_name} already exists. Loading existing data...", log_level="INFO")

        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding,
        )

        return collection

    def _populate_vectorstore(self, collection):
        log("Populating vector store...")
        
        # Prepare data in batches
        batches = [self.df[i:i+self.batch_size] for i in range(0, len(self.df), self.batch_size)]
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for batch in batches:
                future = executor.submit(self._process_batch, batch, collection)
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                future.result()  # This will raise any exceptions that occurred during processing

    def _process_batch(self, batch, collection):
        ids = batch['icd10_code_id'].tolist()
        documents = batch['description'].tolist()
        metadatas = batch.to_dict('records')
        
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def similarity_search_with_score(self, query, top_k):
        return self.vectorstore.similarity_search_with_score(query, top_k)

    def find_top_k_groups(self, results, k):
        seen = set()
        ordered_groups = []
        for result in results:
            icd_code = result[0].metadata['icd10_code_id']
            group = icd_code.split('.')[0]
            if group not in seen:
                seen.add(group)
                ordered_groups.append(group)
                if len(ordered_groups) == k:
                    break
        return ordered_groups

    def filter_df_by_prefixes(self, prefixes):
        return self.df[self.df['icd10_code_id'].apply(lambda x: any(x.startswith(prefix) for prefix in prefixes))]

    def get_articles_by_code(self, icd10_code):
        return list(self.data[self.data['icd10_code_id'] == icd10_code]['article_id'])

    def dataframe_to_list_sorted(self, df):
        return sorted(df.to_dict('records'), key=lambda x: x['icd10_code_id'].strip())

    def get_top_code(self, top_k_prefixes):
        return self.df[self.df['icd10_code_id'].apply(lambda x: any(x.startswith(prefix) for prefix in top_k_prefixes))]        

    def get_description(self, icd10_code_id):
        try:
            result = self.collection.get(ids=[icd10_code_id])['documents'][0]
        except Exception:
            result = None
        return result