import re
import dspy
import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from .vectorstore_manager import VectorStoreManager
from fastapi import HTTPException
import os
from probill.probill.llms import OllamaLocal, DspyModel
from probill.probill.utils.logging_utils import log_debug, log_info, log_error, log
from probill.probill.utils.json_tools import extract_json_objects, try_convert_to_json
from kotaemon.base import Document

current_year = datetime.now().year

_hint_discharge = """
Determine if the service involves discharge management. Evaluate both 'service_description' and 'diagnosis' fields based on the type of service provided:

1. If the service type is 'consultation' or 'consult', discharge management is not included. Therefore, in such cases, return False as discharge management is explicitly excluded.
   Example: Service type is 'consultation' — discharge management is not included.

2. For all other service types, check both 'service_description' and 'diagnosis' for any mentions of 'discharge' or 'discharge management'.
   - If 'discharge' or 'discharge management' is explicitly mentioned in either field, consider that discharge management is included.
   - If there is no mention of 'discharge' or 'discharge management', but the context leaves uncertainty (e.g., ambiguous terms related to patient transition or end of care without clear indication), return True indicating that the presence of discharge management remains unclear.

3. If neither 'discharge' nor 'discharge management' is mentioned and the context does not suggest uncertainty, return False indicating that discharge management is not involved.

The goal is to ensure a clear determination based on the information provided in the 'service_description' and 'diagnosis'. Use this structured approach to analyze and classify each service entry accordingly.
"""

_hint_initiation = """
For each service entry, determine whether it includes an initial evaluation or not.

1. Review both 'diagnosis' and 'service_description' fields for keywords that indicate the nature of the service.
   - If 'follow-up' is mentioned without 'initial evaluation' in either field, classify the service as not having an initial evaluation.
     Example: 'Consultation follow-up' found in 'diagnosis' with no mention of 'initial evaluation' in either field.

2. If 'initial evaluation' is explicitly mentioned in either 'diagnosis' or 'service_description', classify the service as having an initial evaluation.
   Example: 'Initial evaluation with discharge' mentioned in 'diagnosis'.

3. If any service is described as 'X service only' (e.g., 'discharge only', 'consultation only') and there is no reference to 'initial evaluation' or other services in either field, classify the service as not having an initial evaluation.
   Example: 'Discharge only' mentioned in 'diagnosis' with no other services indicated in either field.

Return the classification for each service entry based on the above criteria.
"""

def show_progress(desc: str, title: str = None, channel: str = None):
    # show the evidence
    if not channel:
        channel = "info"
    if not desc:
        yield Document(channel="info", content="<h5><b>No evidence found.</b></h5>")
    else:
        # yield process message
        if title:
            yield Document(
                channel=channel,
                content=(
                    f"<h5>{title}</h5>"
                ),
            )
        if desc:
            yield Document(
                channel=channel,
                content=f"<h5>{desc}</h5>",
            )


class CheckICD10Code(dspy.Signature):
    ("""Only find the ICD10 code written by the physician. Return None if you can not find it, don't make up one by yourself""")
    physician_notes: str = dspy.InputField(desc="the physician's notes")
    icd10_code: str = dspy.OutputField(desc="""Only a JSON object: ```json{"ICD10 Code":"icd10 code or None"}```, no prefix, no additional words""")


class RefineDiagnosis(dspy.Signature):
    ("""Generate concise, complete, formal diagnostic terminology based on the physician's diagnosis""")
    
    physician_notes = dspy.InputField(desc="the physician's diagnosis, notes and medical records for each patient encounter")
    formal_diagnosis = dspy.OutputField(desc="Concise formal diagnostic terminology only based on the physician's diagnosis; don't make it up or predict the ICD 10 code.")

class AssignICD10Code(dspy.Signature):
    ("""Assign ICD 10 code to the diagnosis. Strictly follow the doctor's diagnosis. Did the physician mention a "single episode" or "recurrent"? Don't assume.""")
    
    formal_diagnosis = dspy.InputField(desc="diagnostic terminology")
    icd10_code = dspy.OutputField(desc=""""Only a JSON object: ```json{"ICD10 Code":"icd10 code or None"}```, no prefix, no additional words. Base on the physician's diagnosis strictly; don't make any assumptions by yourself.""")

class  ICD10CodeAssigner(dspy.Module):
    def __init__(self, vec_manager: VectorStoreManager):
        super().__init__()
        self.vec_manager = vec_manager
        self.diagnosis = dspy.ChainOfThought(RefineDiagnosis)
        self.assigner = dspy.ChainOfThoughtWithHint(AssignICD10Code)
        self.check_icd_code = dspy.ChainOfThought(CheckICD10Code)
        self.result = None
        
    def is_icd10_code(self,code):
        """
        Verifies if the provided string is in the ICD-10 code format.
        
        ICD-10 code format:
        - Starts with a letter (A-Z).
        - Followed by two or three digits (0-9).
        - Optionally followed by a period and one or more alphanumeric characters.
        
        Args:
        - code (str): The string to be checked.

        Returns:
        - bool: True if the string is an ICD-10 code, False otherwise.
        """
        # Regex pattern to match ICD-10 codes
        pattern = r'^[A-Za-z][0-9]{2,3}(\.[A-Za-z0-9]+)?$'
        
        # Using re.match to check if the whole string matches the pattern
        if re.match(pattern, code):
            return True
        else:
            return False


    def process_physician_notes(self, physician_notes):
        """Process the input data to extract and format physician notes."""
        physician_notes_json = None

        # Handle different types of input
        if isinstance(physician_notes, list) and physician_notes:
            # Process the first item if it's a list
            physician_notes_json = extract_json_objects(physician_notes[0])
            if not physician_notes_json:
                physician_notes_json = physician_notes[0]
            log("Processed the first item from list.")
        elif isinstance(physician_notes, str):
            # Try to convert string to JSON
            physician_notes_json = extract_json_objects(physician_notes)
            log("Processed string input.")
        elif isinstance(physician_notes, dict):
            # Try to convert string to JSON
            physician_notes_json = physician_notes
            log("Processed string input.")
        
        # Create a formatted string from the JSON object
        if physician_notes_json and isinstance(physician_notes_json, dict):
            # Check if keys exist and handle missing or erroneous data
            service_description = physician_notes_json.get("service_description", "")
            diagnosis = physician_notes_json.get("diagnosis", "").upper()

            diagnosis_str = json.dumps({
                'service_description': service_description,
                'diagnosis': diagnosis
            })
        else:
            # Default to converting the original input to string if JSON conversion fails
            diagnosis_str = physician_notes
            # Log the type and content of the processed data
            log("type: %s", type(diagnosis_str))
            log("Assign ICD10 based on:\n%s", diagnosis_str)

        return diagnosis_str


    def forward(self, physician_notes: str, icd10_data=None):
        
        try:
            
            diagnosis_str = self.process_physician_notes(physician_notes)
            yield from show_progress(f"Start assign ICD code for physician notes:\n{diagnosis_str}")
            
            
            with DspyModel(start_level=0) as models:
                while True:
                    try:
                        if not models.configure_model():
                            break
                        noted_icd = self.check_icd_code(physician_notes=diagnosis_str)
                        break
                    except Exception as e:
                        log_error(str(e))
                        if e == "No more models to configure":
                            break
                                                


            log("noted_icd: %s", noted_icd.icd10_code)
            icd10_code = extract_json_objects(noted_icd.icd10_code)
            log("noted_icd: %s", icd10_code)

            if isinstance(icd10_code, dict):       
                icd10_code = icd10_code.get('ICD10 Code', None).upper()   
                if self.is_icd10_code(icd10_code):
                    icd10_code_desc = self.vec_manager.get_description(icd10_code)
                    if icd10_code_desc:
                        yield from show_progress(f"ICD-10 code: {icd10_code} provided by the physician.")
                        yield from show_progress(f"Describtion: {icd10_code_desc}")
                        self.result = dspy.Prediction(
                            formal_diagnosis="Provided by the physician",
                            icd10_code=icd10_code,
                            icd10_code_desc = self.vec_manager.get_description(icd10_code),
                            rationale="Provided by the physician")
                        return
            
            diagnosis_desc = self.diagnosis(physician_notes=physician_notes)
            docs = self.vec_manager.similarity_search_with_score(diagnosis_desc.formal_diagnosis, 15)
            
            yield from show_progress(f"Similar diagnosis:\n{docs}")
            print("docs:", len(docs))
            
            # Extract the required information and sort it by 'icd10_code_id'
            top_k_prefixes = self.vec_manager.find_top_k_groups(docs,2)
            
            yield from show_progress(f"Sorted:\n{top_k_prefixes}")
            
            # Filter df to include only rows where 'icd10_code_id' starts with any of the prefixes in ordered_groups
            top_df = self.vec_manager.get_top_code(top_k_prefixes)
            # Convert DataFrame to list of dictionaries
            
            print(top_df)

            icd10_code_lst = top_df.to_dict('records')
            icd10_code_lst = sorted(icd10_code_lst, key=lambda x: x['icd10_code_id'].strip())

            # show_progress(f"Candidates:\n{icd10_code_lst}")
            
            # hint=f"""Assign ICD 10 code ID based on the diagnostic terminology\n{icd10_code_lst}"""
            hint=f"""Assign ICD 10 code ID based on the diagnostic terminology\n{docs}"""
            print(hint, flush=True)
            assigned_code = self.assigner(formal_diagnosis=diagnosis_desc.formal_diagnosis, hint=hint)

            # show_progress(f"Assigned ICD-10 code: {assigned_code}")
            icd10_code = extract_json_objects(assigned_code.icd10_code)['ICD10 Code']
            self.result = dspy.Prediction(
                formal_diagnosis=diagnosis_desc.formal_diagnosis,
                icd10_code=icd10_code,
                icd10_code_desc = self.vec_manager.get_description(icd10_code),
                rationale=assigned_code.rationale
            )
        except Exception as e:
            self.result = None
            log_error(f"ERROR 002: {e}")
            raise
            return None
        
class ServiceDates(BaseModel):
    service_dates: Optional[List[str]] = Field(default=None, description="a list of service date, MM/DD/YYYY. could be empty") 
        
class PredictTreatmentActivities(dspy.Signature):
    ("""assign the hcpc code id to the service serial based on the physician's notes""")
    physician_notes = dspy.InputField(desc="the physician's notes and medical records for a service serial")
    icd10_code = dspy.InputField(desc="ICD 10 code id of diagnosis")
    mdm_level = dspy.InputField(desc="determined medical decision level")
    treatment_activities = dspy.OutputField(desc="""a list of objects ordered by the service dates of a service serial\nOnly return a JSON object no prefix, no additional words. ```json{"treatment_activities":[{"service_date": "MM/DD/YYYY", "hcpc_code": "..."}]}```""")

class PredictMedicalDecisionLevel(dspy.Signature):
    ("""Determine the level of Medical Decision Making (MDM) for a given clinical scenario in only one word""")
    physician_notes = dspy.InputField(desc="the physician's notes and medical records of a patient encounter")
    icd10_code = dspy.InputField(desc="ICD 10 code id of diagnosis")
    medical_decision_level = dspy.OutputField(desc='Determined medical decision (MDM) level only one word from the list: ["Low","Moderate","High"]')

class ExtractPlaceOfServiceDates(dspy.Signature):
    ("""Extract service dates from the physician's notes. For each detected date or date range, return the specific dates:
Single Dates: Directly capture any specific single dates mentioned.
Date Ranges: For phrases like "date 1 through date 2", compute and list each date from date 1 to date 2 inclusively.
Return Format: Output the service dates as a list of dates in 'YYYY-MM-DD' format. If no dates are mentioned, return an empty list.
Ensure accurate parsing of dates and date ranges, and handle different formats and synonyms for expressing ranges (e.g., "to", "through", "until").""")    
    physician_notes = dspy.InputField(desc="the physician's notes and medical records of a patient encounter")
    service_dates: ServiceDates = dspy.OutputField(desc=f"List all dates from day one to day x of a date range MM/DD/YY - MM/DD/YY. this year is {current_year}")


class ExtractServiceDates(dspy.Signature):
    ("""Extract service dates from the physician's notes. For each detected date or date range, return the specific dates:
Single Dates: Directly capture any specific single dates mentioned.
Date Ranges: For phrases like "date 1 through date 2", compute and list each date from date 1 to date 2 inclusively.
Return Format: Output the service dates as a list of dates in 'YYYY-MM-DD' format. If no dates are mentioned, return an empty list.
Ensure accurate parsing of dates and date ranges, and handle different formats and synonyms for expressing ranges (e.g., "to", "through", "until").""")    
    physician_notes = dspy.InputField(desc="the physician's notes and medical records of a patient encounter")
    service_dates: ServiceDates = dspy.OutputField(desc=f"List all dates from day one to day x of a date range MM/DD/YY - MM/DD/YY. this year is {current_year}")

class HasDischarge(dspy.Signature):
    ("""Indicate if the service serial has a discharge management""")
    physician_notes = dspy.InputField(desc="the physician's notes and medical records of a patient encounter")
    has_discharge_management: bool = dspy.OutputField(desc='True if the service serial has discharge management on the last day, otherwise False.\nOnly return a bool object no prefix, no additional words.')

class HasInitiation(dspy.Signature):
    ("""Indicate if the service serial has initial evaluation""")
    physician_notes = dspy.InputField(desc="the physician's notes and medical records of a patient encounter")
    has_initial_evaluation: bool = dspy.OutputField(desc="return True if the service serial has initial evaluation. return False if it is a follow-up case.\nOnly return a bool object no prefix, no additional words.")

class ServiceDatesExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_service_dates = dspy.TypedChainOfThought(ExtractServiceDates)

    def process_physician_notes(self, physician_notes):
        """Process the input data to extract and format physician notes."""
        physician_notes_json = None

        # Handle different types of input
        if isinstance(physician_notes, list) and physician_notes:
            # Process the first item if it's a list
            physician_notes_json = extract_json_objects(physician_notes[0])
            log("Processed the first item from list.")
        elif isinstance(physician_notes, str):
            # Try to convert string to JSON
            physician_notes_json = extract_json_objects(physician_notes)
            log("Processed string input.")
        elif isinstance(physician_notes, dict):
            # Try to convert string to JSON
            physician_notes_json = physician_notes
            log("Processed string input.")

        # Log the type and content of the processed data
        log("type: %s", type(physician_notes_json))
        log("Assign ICD10 based on:\n%s", physician_notes_json)
        
        # Create a formatted string from the JSON object
        if physician_notes_json and isinstance(physician_notes_json, dict):
            # Check if keys exist and handle missing or erroneous data
            service_description = physician_notes_json.get("service_description", "")
            diagnosis = physician_notes_json.get("diagnosis", "").upper()

            diagnosis_str = json.dumps({
                'service_description': service_description,
                'diagnosis': diagnosis
            })
        else:
            # Default to converting the original input to string if JSON conversion fails
            diagnosis_str = physician_notes

        return diagnosis_str
            
    def forward(self, physician_notes: str):
        try:
            print("physician_notes\n", physician_notes)
            physician_notes = self.process_physician_notes(physician_notes)
            
            with DspyModel(start_level=0) as models:
                while True:
                    try:
                        if not models.configure_model():
                            break
                        pred = self.extract_service_dates(physician_notes=physician_notes)
                        break
                    except Exception as e:
                        log_error("ServiceDatesExtractor: %s",str(e))     
            
            print(pred)
            service_dates = pred.service_dates.service_dates
            print(service_dates, flush=True)
            return dspy.Prediction(service_dates=service_dates, rationale=pred.reasoning)
        except Exception as e:
            log_error(f"ERROR: {e}")
            return None
            
class PlanningAgent(dspy.Module):
    def __init__(self, hcpc_code_json, mdm_level_json, num_preds = 1, initiation_promp:str=None, ):
        super().__init__()
        self.hcpc_code_json = hcpc_code_json
        self.mdm_level_json = mdm_level_json
        self.mdm_level_determine = dspy.ChainOfThoughtWithHint(PredictMedicalDecisionLevel, n=num_preds)
        self.predict = dspy.ChainOfThoughtWithHint(PredictTreatmentActivities, n=num_preds)
        self.has_discharge = dspy.ChainOfThoughtWithHint(HasDischarge, n=num_preds)
        self.has_initiation = dspy.ChainOfThoughtWithHint(HasInitiation, n=num_preds)
        self.result = None
        self.initiation_promp = 
        
    def forward(self, physician_notes: str, icd10_code: str):
        hint_1=f"""Determine the level of Medical Decision Making (MDM) for a given clinical scenario by evaluating the following criteria:.\n{self.mdm_level_json}.\nThe most of psychological Medical decision making (MDM) are level high, especially with a diagnosis of severe; Follow the MDM level if it is explicitly mentioned in 'service_description' or 'diagnosis'."""
        
        try:
            yield from show_progress(f"Start determine the mdm level: {icd10_code}")
            mdm_level = self.mdm_level_determine(physician_notes=physician_notes, icd10_code=icd10_code, hint=hint_1)
            
            hint_discharge = _hint_discharge
            hint_initiation = _hint_initiation
            has_discharge_management = self.has_discharge(physician_notes=physician_notes, hint = hint_discharge)
            has_initiation = self.has_initiation(physician_notes=physician_notes, hint = hint_initiation)
            yield from show_progress(f"has_initial_evaluation: {has_initiation.has_initial_evaluation}")
            yield from show_progress(f"has_discharge_management: {has_discharge_management.has_discharge_management}")
            
            initiation_promp =[
                f'assign only initial code on the first day, assign follow-up code for the rest dates with {mdm_level.medical_decision_level} MDM level',
                f'it is a follow-up case, do not assign initial code to the first day, assign follow-up code to the first and rest service dates with {mdm_level.medical_decision_level} MDM level'                
            ]
            
            discharge_promp = [
                f'Do not use "Services Including Admit & Discharge at the same day" unless it is a one day service.\n3. Assign a code of "Discharge Day Management" to the last day only with {mdm_level.medical_decision_level} MDM level.',
                'do not assign a code of "Discharge Day Management" to the last day of the service serial'
            ]
            
            hint_2=f"""Refer to the following JSON object, which lists candidate treatment activities and determines the medical decision level.\n{self.hcpc_code_json}. \nFill in the 'service_date' for each step with the appropriate dates. Assign at least one activity every day. You can apply one activity to multiple days. Follow the principles below:\n1. {initiation_promp[0] if has_initiation.has_initial_evaluation=='True' else initiation_promp[1]};\n2. {discharge_promp[0] if has_discharge_management.has_discharge_management=='True' else discharge_promp[1]}."""
                
            yield from show_progress(f"Start assign HCPC code for ICD 10 CODE: {icd10_code} on MDM level: {mdm_level.medical_decision_level}")
            yield from show_progress("Rationale:")
            yield from show_progress(mdm_level.rationale)
            log_info("physician_notes: %s", physician_notes)
            
            # prediction = self.predict(physician_notes=physician_notes, icd10_code=icd10_code, mdm_level=mdm_level.medical_decision_level, hint=hint_2)
            prediction = self.predict(physician_notes=physician_notes, icd10_code=icd10_code, mdm_level="High", hint=hint_2)
            yield from show_progress(f"treatment_activities:\n{prediction.treatment_activities}")
            self.result = dspy.Prediction(treatment_activities=extract_json_objects(prediction.treatment_activities), rationale=prediction.rationale)
            return True
        except Exception as e:
            log_error(f"ERROR: {e}")
            raise
            return False
        
class CodeAgent:
    def __init__(self, database_path=None):
        self.results = None

        # Get the database name from environment variables
        if not database_path:
            database_path = os.getenv('DB_PATH', '../database/')
            
        # database_path = "/app/taskflow/db/"
        # database_path = "../../agents/db/"
        hcpc_code_path = f"{database_path}current_article_csv/locations.json"
        mdm_level_path = f"{database_path}current_article_csv/mdm_level.json"
        with open(hcpc_code_path, "r") as f:
            hcpc_code_json = json.load(f)

        with open(mdm_level_path, "r") as f:
            mdm_level_json = json.load(f)

        self.vec_manager = VectorStoreManager(f"{database_path}/current_article_csv/article_x_icd10_covered.csv")
        self.planner = PlanningAgent(hcpc_code_json=hcpc_code_json,mdm_level_json=mdm_level_json)
        self.icd10_assigner = ICD10CodeAssigner(vec_manager=self.vec_manager)
        self.service_dates_extractor = ServiceDatesExtractor()
    
    def sort_and_format_dates(self, date_list):
        def parse_date(date):
            # Add the new format to the list of formats to be tried
            for fmt in ('%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d'):
                try:
                    return datetime.strptime(date, fmt)
                except ValueError:
                    continue
            # If no format matches, log or handle the specific error case here
            return None
        
        # Convert the dates to datetime objects, filtering out None values for unsupported formats
        date_objects = [parse_date(date) for date in date_list if parse_date(date) is not None]
        
        # Sort the datetime objects
        sorted_dates = sorted(date_objects)
        
        # Convert the sorted datetime objects back to a consistent format
        formatted_dates = [date.strftime('%m/%d/%Y') for date in sorted_dates]
        
        return formatted_dates
    
    def process_physician_notes(self, physician_notes):
        """Process the input data to extract and format physician notes."""
        physician_notes_json = None

        # Handle different types of input
        if isinstance(physician_notes, list) and physician_notes:
            # Process the first item if it's a list
            physician_notes_json = extract_json_objects(physician_notes[0])
            log("Processed the first item from list.")
        elif isinstance(physician_notes, str):
            # Try to convert string to JSON
            physician_notes_json = extract_json_objects(physician_notes)
            log("Processed string input.")
        elif isinstance(physician_notes, dict):
            # Try to convert string to JSON
            physician_notes_json = physician_notes
            log("Processed string input.")

        # Log the type and content of the processed data
        log("type: %s", type(physician_notes_json))
        log("Assign ICD10 based on:\n%s", physician_notes_json)
        
        # Create a formatted string from the JSON object
        if physician_notes_json and isinstance(physician_notes_json, dict):
            # Check if keys exist and handle missing or erroneous data
            service_description = physician_notes_json.get("service_description", "")
            diagnosis = physician_notes_json.get("diagnosis", "").upper()

            diagnosis_str = json.dumps({
                'service_description': service_description,
                'diagnosis': diagnosis
            })
        else:
            # Default to converting the original input to string if JSON conversion fails
            diagnosis_str = physician_notes

        return diagnosis_str
                    
    def code(self, physician_notes: str, model=None):
        # if not isinstance(physician_notes,list):
        #     physician_notes = [physician_notes]
        yield from self.icd10_assigner(physician_notes)
        # physician_notes = self.process_physician_notes(physician_notes)
        icd10_result = self.icd10_assigner.result
        icd10_code = json.dumps({
            'icd10_code':icd10_result.icd10_code,
            'icd10_code_desc': icd10_result.icd10_code_desc
        })

        yield from show_progress(icd10_code)
        
        if icd10_result is None:
            yield from show_progress("ICD-10 code not found for the given notes")
            raise HTTPException(status_code=404, detail="ICD-10 code not found for the given notes")
        else:
            physician_notes = json.loads(physician_notes)
            if not isinstance(physician_notes, list):
                physician_notes = [physician_notes]
            for physician_note in physician_notes:
                service_dates = physician_note.get('service_dates', None)
                if len(physician_note['service_dates']) == 0 or physician_note['service_dates'] == "":
                    service_dates = self.service_dates_extractor(physician_note)
                    physician_note['service_dates'] = service_dates.service_dates
                    
                if len(physician_note['service_dates']) > 0:
                    # Split the string into a list of date strings
                    
                    if isinstance(physician_note['service_dates'], str):
                        date_list = physician_note['service_dates'].split(', ')
                    elif isinstance(physician_note['service_dates'], list):
                        date_list = physician_note['service_dates']
                        
                    # Convert string dates to datetime objects and sort them
                    sorted_dates = self.sort_and_format_dates(date_list)
                    yield from show_progress(f"sorted_dates:\n{sorted_dates}")

                    # Convert the list back to a single string if needed
                    physician_note['service_dates'] = ', '.join(sorted_dates)
                    
                    yield from self.planner(physician_notes=f'{physician_note}', icd10_code=icd10_code)
                    hcpc_result = self.planner.result
                    
                    if hcpc_result is None:
                        raise HTTPException(status_code=404, detail="hcpc code not found for the given notes")
                    else:
                        hcpc_result_json = None
                        if isinstance(hcpc_result["treatment_activities"], str):
                            try:
                                hcpc_result_json = try_convert_to_json(hcpc_result.treatment_activities)
                            except Exception as e:
                                yield from show_progress(f"Cannot load hcpc_result_json to a JSON:\n{hcpc_result_json} with error: str{e}")
                                hcpc_result_json = hcpc_result.treatment_activities
                        elif isinstance(hcpc_result["treatment_activities"], dict):
                            hcpc_result_json = hcpc_result.treatment_activities
                            
                        yield from show_progress("hcpc_result_json:\n{hcpc_result_json}", channel="debug")
                        
                        yield from show_progress(
                            json.dumps({
                            "icd10_code": icd10_result.icd10_code,
                            "icd10_code_desc": icd10_result.icd10_code_desc,
                            "hcpc_result": hcpc_result_json
                            }, indent=4),
                            channel = "chat"                    
                        )
                else:
                    yield from show_progress(
                        json.dumps({
                            "icd10_code": icd10_result.icd10_code,
                            "icd10_code_desc": icd10_result.icd10_code_desc,
                            "hcpc_result": "no service dates found"
                        }, indent=4),
                        channel = "chat"                    
                    )
