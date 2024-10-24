import re
import json
import base64
import io
import hashlib
from PIL import Image
from pydantic import BaseModel, create_model
from typing import Any, Dict, Type, List, Union

def encode_image_to_base64(image) -> str:
    """
    Encode an image to base64.

    Args:
        image (PIL.Image.Image): The image to encode.

    Returns:
        str: Base64 encoded image string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_image(image_str)-> Image.Image:
    """
    Convert a base64-encoded image back to a PIL Image object.

    Args:
        base64_encoded_image (str): The base64-encoded image string.

    Returns:
        PIL.Image: The decoded Image object.
    """
    # Decode the base64 string
    decoded_image = base64.b64decode(image_str)

    # Convert the bytes to a BytesIO object
    image_stream = io.BytesIO(decoded_image)

    # Load the BytesIO object into a PIL Image
    return Image.open(image_stream)

def extract_json_objects(text):
    """
    Extracts a JSON object or array from a given string.
    
    Args:
    - text (str): The string containing a JSON object or array.
    
    Returns:
    - Python object: A dictionary or list extracted from the JSON string found in the text.
                     Returns None if no valid JSON structure is found.
    """
    try:
        # Remove markdown code block identifiers if present
        cleaned_text = re.sub(r'```json\n|```', '', text, flags=re.DOTALL).strip()

        # Try to load the JSON data directly from the cleaned text
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # If direct loading fails, try to find JSON objects using regex
        json_str = re.search(r'(\{.*\}|\[.*\])', cleaned_text, re.DOTALL)
        
        if json_str:
            try:
                return json.loads(json_str.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

def hash_object(obj:Union[str, Dict]):
    """
    Hash an object by serializing it to a JSON string, encoding it to bytes, and hashing the bytes.
    """
    try:
        # Serialize the object to a JSON string
        if isinstance(obj, str):
            json_str = obj
        else:
            json_str = json.dumps(obj, sort_keys=True)
        
        # Encode the JSON string to bytes
        json_bytes = json_str.encode('utf-8')
        
        # Compute the hash of the bytes
        obj_hash = hashlib.md5(json_bytes).hexdigest()
        
        return obj_hash
    except TypeError as e:
        # Handle the error if the object cannot be serialized to JSON
        print(f"Failed to hash object: {e}")
        return None

def try_convert_to_json(value):
    """Try to convert a value to JSON."""
    if isinstance(value, bytes):
        value = value.decode('utf-8')  # Decode bytes to string
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return value
    elif isinstance(value, dict):
        return {k: try_convert_to_json(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [try_convert_to_json(v) for v in value]
    else:
        return value
    
def generate_pydantic_model(name: str, data: Dict[str, Any]) -> Type[BaseModel]:
    def parse_field(value: Any):
        if isinstance(value, bool):
            return (bool, ...)
        elif isinstance(value, int):
            return (int, ...)
        elif isinstance(value, float):
            return (float, ...)
        elif isinstance(value, str):
            return (str, ...)
        elif isinstance(value, list):
            if value:
                return (List[parse_field(value[0])[0]], ...)
            else:
                return (list, ...)
        elif isinstance(value, dict):
            return (Dict[str, Any], ...)
        else:
            return (Any, ...)
    
    fields = {k: parse_field(v) for k, v in data.items()}
    return create_model(name, **fields)


def find_key_path(d, target_key):
    """
    Find the path to the given key in a nested dictionary.

    Parameters:
    - d (dict): The dictionary to search.
    - target_key (str): The key to find.

    Returns:
    - list: The path to the key as a list of keys, or an empty list if key is not found.
    """
    # Helper function to perform a recursive search
    def search(subdict, path):
        if isinstance(subdict, dict):
            if target_key in subdict:
                return path + [target_key]
            for key, value in subdict.items():
                result_path = search(value, path + [key])
                if result_path:
                    return result_path
        elif isinstance(subdict, list):
            for index, item in enumerate(subdict):
                result_path = search(item, path + [index])
                if result_path:
                    return result_path
        return []

    # Start the search from the root of the dictionary
    return search(d, [])

def get_value_by_path(d, path):
    """
    Retrieve the value from a nested dictionary based on the given path.
    
    Parameters:
    - d (dict): The root dictionary from which to retrieve the value.
    - path (list): A list of keys and indices representing the path to the target value.
    
    Returns:
    - The value found at the specified path, or None if the path is invalid.
    """
    current = d
    try:
        for key in path:
            current = current[key]
        return current
    except (KeyError, IndexError, TypeError):
        return None  # Path is invalid or the structure isn't as expected

# def get_value_by_key(d, target_key):
#     """
#     Retrieve the value from a nested dictionary based on the given key.
    
#     Parameters:
#     - d (dict): The root dictionary from which to retrieve the value.
#     - target_key (str): The key to find and retrieve the value for.
    
#     Returns:
#     - The value found for the specified key, or None if the key is not found.
#     """
#     # Helper function to perform a recursive search
#     def search(subdict, path):
#         if isinstance(subdict, dict):
#             if target_key in subdict:
#                 return path + [target_key]
#             for key, value in subdict.items():
#                 result_path = search(value, path + [key])
#                 if result_path:
#                     return result_path
#         elif isinstance(subdict, list):
#             for index, item in enumerate(subdict):
#                 result_path = search(item, path + [index])
#                 if result_path:
#                     return result_path
#         return []

#     # Find the path to the key
#     path = search(d, [])

#     # Retrieve the value by path if it exists
#     if path:
#         current = d
#         try:
#             for key in path:
#                 current = current[key]
#             return current
#         except (KeyError, IndexError, TypeError):
#             return None  # Path is invalid or the structure isn't as expected
#     else:
#         return None  # Key not found

# class MultipleMatchesError(Exception):
#     def __init__(self, *args, matches=None, **kwargs):
#         self.matches = matches
#         super().__init__(*args, **kwargs)

def get_value_by_key(d, target_key_expression, default_value=None, multiple_matches=False, first_match=True):
    """
    Retrieve the value from a nested dictionary based on the given key expression.
    
    Parameters:
    - d (dict): The root dictionary from which to retrieve the value.
    - target_key_expression (str): The key expression to find and retrieve the value for, formatted as "a->b->c", "a.b.c", or "a-b-c".
    - default_value: The value to return if the key expression is not found. Default is None.
    - multiple_matches (bool): Whether to allow multiple matches. Default is False.
    - first_match (bool): Whether to return the first match if multiple matches are found and allowed. Default is True.
    
    Returns:
    - The value found for the specified key expression.
    
    Raises:
    - MultipleMatchesError: If multiple matches are found for the specified key expression and multiple_matches is False.
    - KeyError: If the key expression is not found.
    """
    # Normalize the target_key_expression to use a standard delimiter
    normalized_keys = target_key_expression.replace(" ", "").replace("->", ".").replace("-", ".").split('.')
    matches = []

    def search(subdict, keys):
        if not keys:
            matches.append(subdict)
            return
        key = keys[0]
        if isinstance(subdict, dict):
            if key in subdict:
                search(subdict[key], keys[1:])
            else:
                for k, v in subdict.items():
                    if isinstance(v, dict):
                        search(v, keys)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                search(item, keys)
        elif isinstance(subdict, list):
            for item in subdict:
                if isinstance(item, dict):
                    search(item, keys)
        return
    
    search(d, normalized_keys)

    if len(matches) > 1:
        if multiple_matches:
            if first_match:
                return matches[0]
            else:
                return matches
        else:
            raise MultipleMatchesError("Multiple matches found", matches=matches)
    elif len(matches) == 0:
        return default_value
    else:
        return matches[0]

# Custom error class for multiple matches
class MultipleMatchesError(Exception):
    def __init__(self, message, matches=None):
        super().__init__(message)
        self.matches = matches if matches else []

