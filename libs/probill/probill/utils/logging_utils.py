# logging_utils.py
import logging
import json
import os
import copy
import inspect
from .json_tools import try_convert_to_json

# ANSI escape sequences for coloring
class LogColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'      # Use blue for general information
    OKCYAN = '\033[96m'      # Cyan color for DEBUG (more readable)
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class CustomFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: LogColors.OKCYAN,
        logging.INFO: LogColors.OKGREEN,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.FAIL,
        logging.CRITICAL: LogColors.FAIL + LogColors.BOLD
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        # Base format for log output
        base_format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        # Apply color to the message part only
        formatter = logging.Formatter(base_format)
        formatted_message = formatter.format(record)
        # Split the message into lines and apply color to each line
        lines = formatted_message.split('\n')
        colored_lines = [log_fmt + line + LogColors.ENDC for line in lines]
        colored_message = '\n'.join(colored_lines)
        return colored_message
    
# Configure the logging
logger = logging.getLogger("probill_logger")
logger.propagate = False  # Prevent propagation to the root logger

# Remove all existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Setting the custom formatter
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# Map string log levels to logging constants
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "NOTSET": logging.NOTSET  # Added "NOTSET" as a way to disable logging
}

# Get the LOG_LEVEL from environment, default to INFO if not set
env_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = LOG_LEVEL_MAP.get(env_log_level, logging.INFO)

# If the log level is "NOTSET", effectively disable logging
logger.setLevel(log_level)

def shorten_long_items(obj, max_items=10, disp_items=5, max_len=500, disp_len=500, depth=2):
    if isinstance(obj, dict):
        if depth <= 0:
            return f"{{ ... {len(obj)} items ... }}"
        return {k: shorten_long_items(v, max_items, disp_items, max_len, disp_len, depth-1)
                for k, v in list(obj.items())[:max_items]}

    elif isinstance(obj, list):
        if depth <= 0 or len(obj) > max_items:
            return [shorten_long_items(i, max_items, disp_items, max_len, disp_len, depth-1) for i in obj[:disp_items]] + (["..."] if len(obj) > disp_items else [])
        return [shorten_long_items(i, max_items, disp_items, max_len, disp_len, depth-1) for i in obj]

    elif isinstance(obj, str) and len(obj) > max_len:
        return obj[:disp_len] + " ... " + obj[-disp_len:]

    elif isinstance(obj, bytes):
        try:
            obj = obj.decode("utf-8")
        except UnicodeDecodeError:
            return "<binary data>"
        return shorten_long_items(obj, max_items, disp_items, max_len, disp_len, depth)

    return obj

def process_args(args, max_items=15, disp_items=5, max_len=300, disp_len=200, depth=3):
    """Process each argument, applying shorten_long_items to dictionaries."""
    processed_args = []
    for arg in args:
        arg_copy = copy.deepcopy(arg)
        arg_copy = try_convert_to_json(arg_copy)
        arg_copy = shorten_long_items(arg_copy, max_items, disp_items, max_len, disp_len, depth)
        if isinstance(arg_copy, dict) or isinstance(arg_copy, list):
            try:
                arg_copy = json.dumps(arg_copy, indent=2)
            except:
                pass
            processed_args.append(arg_copy)
        else:
            processed_args.append(arg_copy)
    return processed_args

# def log(message, *args, log_level="DEBUG", max_items=5, disp_items=5, max_len=50, disp_len=10, depth=2):
#     # Handle non-string messages
#     if not isinstance(message, str):
#         return log("%s", message, *args, log_level=log_level, max_items=max_items, disp_items=disp_items, max_len=max_len, disp_len=disp_len, depth=depth)
    
#     if isinstance(log_level, str):
#         log_level = LOG_LEVEL_MAP.get(log_level.upper(), logging.DEBUG)
    
#     if log_level < logger.level:
#         return

#     # Adjust parameters for DEBUG level
#     if log_level == logging.DEBUG:
#         max_items, disp_items, max_len, disp_len, depth = 30, 5, 300, 20, 5

#     # Get the correct caller information for logging
#     current_frame = inspect.currentframe()
#     caller_frame = None
#     try:
#         frame = current_frame
#         while frame:
#             if frame.f_code.co_name != 'log':
#                 caller_frame = frame
#                 break
#             frame = frame.f_back
        
#         if caller_frame:
#             fn = caller_frame.f_code.co_filename
#             lno = caller_frame.f_lineno
#             func = caller_frame.f_code.co_name
#         else:
#             fn, lno, func = "(unknown file)", 0, "(unknown function)"
#     finally:
#         del current_frame
#         del caller_frame

#     ch.setFormatter(CustomFormatter())
#     processed_args = process_args(args, max_items, disp_items, max_len, disp_len, depth)
#     if args and '%' in message:
#         try:
#             message = message % tuple(processed_args)
#         except TypeError:
#             log(logging.WARNING, "log received mismatched format string and arguments.")
#             message += " [Args: " + ", ".join(map(str, args)) + "]"
#     else:
#         message = shorten_long_items(message, max_items=max_items, disp_items=disp_items, max_len=max_len, disp_len=disp_len, depth=depth)

#     try:
#         record = logger.makeRecord(logger.name, log_level, fn, lno, message, processed_args, None, func)
#         logger.handle(record)
#     except Exception as e:
#         print(f"Error in log: {e}")
#         if 'record' in locals():
#             print(f"Record details: {record.__dict__}")

def log(message, *args, log_level="DEBUG", max_items=8, disp_items=5, max_len=300, disp_len=150, depth=2, through_caller=False):
    # Handle non-string messages
    if not isinstance(message, str):
        return log("%s", message, *args, log_level=log_level, max_items=max_items, disp_items=disp_items, max_len=max_len, disp_len=disp_len, depth=depth)
    
    message_copy = copy.deepcopy(message)
    
    if isinstance(log_level, str):
        log_level = LOG_LEVEL_MAP.get(log_level.upper(), logging.DEBUG)
    
    if log_level < logger.level:
        return

    # Adjust parameters for DEBUG level
    if log_level == logging.DEBUG:
        max_items, disp_items, max_len, disp_len, depth = 30, 5, 1000, 20, 5

    # Get the correct caller information for logging
    if through_caller:
        caller_frame = inspect.stack()[2]  # [2] gives us the caller of the logging function (log_debug, log_info, etc.)
    else:
        caller_frame = inspect.stack()[1]  # [2] gives us the caller of the logging function (log_debug, log_info, etc.)
    fn = caller_frame.filename
    lno = caller_frame.lineno
    func = caller_frame.function        

    ch.setFormatter(CustomFormatter())
    processed_args = process_args(args, max_items, disp_items, max_len, disp_len, depth)
    if args and '%' in message_copy:
        try:
            message_copy = message_copy % tuple(processed_args)
        except TypeError:
            log(logging.WARNING, "log received mismatched format string and arguments.")
            message_copy += " [Args: " + ", ".join(map(str, args)) + "]"
    else:
        message_copy = shorten_long_items(message_copy, max_items=max_items, disp_items=disp_items, max_len=max_len, disp_len=disp_len, depth=depth)

    try:
        record = logger.makeRecord(logger.name, log_level, fn, lno, message_copy, processed_args, None, func)
        logger.handle(record)
    except Exception as e:
        print(f"Error in log: {e}")
        if 'record' in locals():
            print(f"Record details: {record.__dict__}")

def format_json_str(jstr):
    """Remove newlines outside of quotes, and handle JSON escape sequences."""
    result = []
    inside_quotes = False
    last_char = " "
    for char in jstr:
        if last_char != "\\" and char == '"':
            inside_quotes = not inside_quotes
        last_char = char
        if not inside_quotes and char == "\n":
            continue
        if inside_quotes and char == "\n":
            char = "\\n"
        if inside_quotes and char == "\t":
            char = "\\t"
        result.append(char)
    return "".join(result)

# Example usage
def log_debug(message: str, *args, **kwargs):
    log(message, *args, **kwargs, log_level="DEBUG", through_caller=True)

def log_info(message: str, *args, **kwargs):
    log(message, *args, **kwargs, log_level="INFO", through_caller=True)

def log_warning(message: str, *args, **kwargs):
    log(message, *args, **kwargs, log_level="WARNING", through_caller=True)

def log_error(message: str, *args, **kwargs):
    log(message, *args, **kwargs, log_level="ERROR", through_caller=True)

def log_dict_as_json(dict_obj: dict, log_function):
    """Logs a dictionary as a JSON string using the specified logging function."""
    if logger.level != logging.NOTSET:
        ch.setFormatter(CustomFormatter())
        shortened_dict = shorten_long_items(dict_obj)
        log_function(json.dumps(shortened_dict, indent=2))