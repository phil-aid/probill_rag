
import dspy
import os
from .ollama import OllamaLocal
from probill.probill.utils.logging_utils import log_error, log_warning, log_info

class DspyModel:
    def __init__(self, model_name=None, start_level=0):        
        # set DSPy to use gpt-3.5
        nvidia_base_url = "http://10.0.40.49:11434"
        mac_base_url = "http://10.0.40.244:11434"
        ollama_model = model_name if model_name else "gemma2:9b-instruct-fp16"
        mac_model = "llama3.1:8b-instruct-fp16"
        self.start_level = start_level
        api_key = os.getenv('API_KEY', 'sk-xxx')

        
        
        mac_lm = OllamaLocal(
            model=mac_model,
            base_url=mac_base_url,
            temperature=0.0,
            top_k=20,
            top_p=0.5,
            num_ctx=2048,
            max_tokens=4096,
            timeout_s=500,
        )

        nvidia_lm = OllamaLocal(
            model=ollama_model,
            base_url=nvidia_base_url,
            temperature=0.0,
            top_k=20,
            top_p=0.5,
            num_ctx=2048,
            max_tokens=4096,
            timeout_s=500,
        )

        local_llama_31 = OllamaLocal(
            model="llama3.1:8b-instruct-fp16",
            base_url=nvidia_base_url,
            temperature=0.0,
            top_k=20,
            top_p=0.5,
            num_ctx=2048,
            max_tokens=4096,
            timeout_s=500,
        )

        openai_lm = dspy.OpenAI(
            model='gpt-4o-mini-2024-07-18', 
            api_key=api_key, 
            temperature=0.0, 
            top_p=0.5,
            # max_tokens=1024
        )
                
        self.llms = {
            0: nvidia_lm,
            # 1: mac_lm,
            2: openai_lm
        }

    def __enter__(self):
        self.level = self.start_level
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def configure_model(self):
        if self.level < len(self.llms):
            log_info(f"Model level: {self.level}")
            dspy.settings.configure(lm=self.llms[self.level])
            self.level += 1
            return True
        else:
            log_error("No more models to configure")
            return False

    def process_with_callback(self, callback, **kwargs):
        while True:
            try:
                if self.configure_model():
                # Use the callback function with the configured model
                    result = callback(**kwargs)
                    return result
                else:
                    return False
            except Exception as e:
                log_error(str(e))
                log_warning("Trying next model level")


