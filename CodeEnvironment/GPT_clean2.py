# #Uncomment in third-party IDE
# allow_run = True

if allow_run:

    # -----------------------*** IMPORT MODULES ***-----------------------

    ## Uncomment in GH (rhinoscriptsyntax can only run in GH python component)
    import rhinoscriptsyntax as rs


    # GENERAL MODULES
    import logging
    import os
    import shutil
    import sys
    import json_repair
    import json
    import glob
    import time
    import re
    from typing import List
    from dataclasses import dataclass
    from typing import Optional, Any
    from pathlib import Path
    import sys


    # LLM MODULES
    
    # Claude
    from anthropic import Anthropic

    # ChatGPT
    import openai
    
    # Mistral
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage

    # Google Gemini (not currently in use)
    from google.cloud import storage
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part, Content, FunctionDeclaration, Tool
    from vertexai.generative_models._generative_models import ToolConfig
    import google.ai.generativelanguage as glm

    # Meta LLaMa (potentially)
  


    # -----------------------*** VARIABLE AND PATH INITILIZATION ***-----------------------

    # # Uncomment if editing in third-party IDE, otherwise inputs directly from input block in GH
    # go = True
    # reset = False
    # FinishedCalled = False
    # Allow_askback = False

    # GPT = "ChatGPT"
    # OpenAI_model = "gpt-4o"
    # prompt_name ="toycar_a"

    # # replace path with location of CodeEnvironment folder. CodeEnvironment contains:
    #     # functions folder: block.json, cylinder.json, finished.json, force.json, CreateCode.json
    #     # temp folder: instances of functions (JSON files), crash_file.txt, terminal_output.txt
    #     # keys for ChatGP (key.txt), Anthropic, and Mistral
    #     # geometry_creation.py: module containing function defintions for functions that parse JSON object instances into rhinoscriptsyntax 
    #         # functions, imported into CreateGeometries python block in GH (Note: brings up the question of why that is not done in GPT 
    #         # python block as well.
    #     # log.json: contains log of conversation with LLM, including the system definition, prompt, and function instantiations
    #     # task.json: contains the descriptions of the system and each agent workflow (askBack, stepPlanning, visualInspection)
    #     # test_prompts.json: contains text prompts
        
    # path = r"P:\01_Research_projects\2023_DNT_LLM_in_Engineering_Applications\04_Paper\Code\Grasshopper\CodeEnvironment"

    # # replace base_destination_path with location of folder where results will be saved. These include:
    #     # temp folder
    #     # log.json
    #     # STL file (if exported)

    # base_destination_path = r"P:\01_Research_projects\2023_DNT_LLM_in_Engineering_Applications\04_Paper\Experiments\Different GPT\ChatGPT"

    ## Uncomment until here

    # number attached to end of new_folder_name will be increased automatically
    new_folder_name= "TestPrompts"

    logfile = "log.json"
    taskfile = "task.json"

    debug = True
    change = False
    click = False
    click_filename = "Click"

    # Set LLM functions relative path 
    directory_path = "functions/"  # Adjust the path to your functions directory

    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    os.chdir(path)

    sys.path.append

    # Create temp folder
    folder = "temp"
    # Relative terminal path
    terminal_path = os.path.join(folder, "terminal_output.txt")
    # Absolute terminal path
    gh_terminal_path = os.path.join(path, terminal_path)
   
    text = [] #Does not appear to be necessary, consider removing

    # All inputs and responses are appended to messages and periodically logged onto log.json
    messages = []

    # -----------------------*** FUNCTIONS ***-----------------------
    # Note: Should be found in a separate module and then imported with something like this:
    # from gpt_functions import *, but currently many function definitions depend on variables in the main program 
    # that are not passed to the function definitons, so simply copy and pasting these functions onto a separarte
    # module is complicated


    # LOGGING
    class StreamToLogger:
        """
        Custom stream object that redirects writes to a logger instance.
        """
        def __init__(self, logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level
            self.linebuf = ''

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())

        def flush(self):
            pass

    # Set up logging
    logging.basicConfig(filename=terminal_path,
                        filemode='w',
                        format='%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger()

    # Redirect stdout to the logging handler
    sys.stdout = StreamToLogger(logger, logging.INFO)

    def logger(messages, logfile):
        with open(logfile, "w") as tf:
            # Use the default parameter to specify your conversion function
            json.dump(messages, tf, default=default_converter)


    # FORMATTING AND DATA TRANSFER
    def custom_serializer(obj):
        if hasattr(obj, 'fields'):
            # Assumes fields is a dictionary of key-value pairs
            return {k: custom_serializer(v) for k, v in obj.fields.items()}
        elif hasattr(obj, 'number_value'):
            # Direct conversion of number_value
            return obj.number_value
        elif hasattr(obj, 'struct_value'):
            # Recursive handling of struct_value
            return custom_serializer(obj.struct_value)
        elif hasattr(obj, 'list_value'):
            # Handle list values
            return [custom_serializer(item) for item in obj.list_value]
        else:
            # Check for other types that json can serialize directly
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                # Final catch for any types that are still not handled
                return str(obj)

    def parse_list(arguments):
        # Filter out non-list values and empty lists
        list_values = [v for v in arguments.values() if isinstance(v, list)]
        print("--------------------------Parsing Function Arguments -------------------------------")
        if not list_values:
            # If there are no lists, return original arguments
            print("No lists found in the arguments.")
            return arguments
        
        # Determine the maximum length among all non-empty lists
        max_length = max(len(v) for v in list_values)

        # Ensure boolean lists remain as lists of booleans
        for key, value in arguments.items():
            if isinstance(value, list) and all(isinstance(item, bool) for item in value):
                print(f"Ensuring boolean list for key '{key}' remains as booleans.")
                value[:] = [bool(item) for item in value]
        
        # Iterate through each key and update lists as needed
        for key, value in arguments.items():
            if isinstance(value, list):
                current_length = len(value)
                if current_length < max_length:
                    if current_length == 0:
                        value.extend([0] * max_length)
                        print(f"\033[91mFilled '{key}' with {max_length} zeros.\033[0m")
                    else:
                        last_element = value[-1]
                        additional_elements = max_length - current_length
                        value.extend([last_element] * additional_elements)
                        print(f"Added {additional_elements} elements with value '{last_element}' to '{key}'.")
                elif current_length == 0:
                    value.extend([0] * max_length)
                    print(f"\033[91mFilled '{key}' with {max_length} zeros.\033[0m")
        
        return arguments

    def create_unique_folder(base_path, base_name):
        # Start with index 1 since the first folder should be base_name_1
        index = 1
        while True:
            new_folder_name = f"{base_name}_{index}"
            new_folder_path = os.path.join(base_path, new_folder_name)
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
                print(f"Created directory: {new_folder_path}")
                return new_folder_path
            index += 1

    def copy_files_and_folders(base_destination, folder_name, destination = None):
        # Create a unique folder inside the destination
        if destination is None:
            destination = create_unique_folder(base_destination, folder_name)
        
        # List of items to copy
        items_to_copy = ['temp', 'log.json']
        
        for item in items_to_copy:
            # Construct source path
            source_path = os.path.join(os.getcwd(), item)
            # Construct destination path within the newly created unique folder
            dest_path = os.path.join(destination, item)
            
            # Check if the item exists before attempting to copy
            if os.path.exists(source_path):
                if os.path.isdir(source_path):
                    # Copy the directory recursively
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True) 
                    print(f"Copied {item} to {destination}")
                else:
                    # Copy the file
                    shutil.copy(source_path, dest_path)
                    print(f"Copied {item} to {destination}")
            else:
                print(f"{item} does not exist in the source directory.")
        return destination

    def repair_python_to_JSON(json_like_string):
        # Temporarily replace escaped single quotes to avoid conversion
        temp_placeholder = re.sub(r"\\'", "TEMP_SINGLE_QUOTE", json_like_string)
        
        # Convert single quotes to double quotes
        repaired_string = temp_placeholder.replace("'", '"')
        
        # Revert temporarily replaced escaped single quotes back to original
        repaired_string = re.sub(r"TEMP_SINGLE_QUOTE", "\\'", repaired_string)
        
        # Replace Python's True and False with JSON's true and false
        repaired_string = re.sub(r"\bTrue\b", "true", repaired_string)
        repaired_string = re.sub(r"\bFalse\b", "false", repaired_string)
        
        # Replace Python's None with JSON's null
        repaired_string = re.sub(r"\bNone\b", "null", repaired_string)
        
        return repaired_string

    def sanitize_subtraction_field(directory: str) -> None:
        """
        Sanitizes the 'subtraction' field in all .json files within the given directory.
        Args:
        - directory (str): The path to the directory containing the .json files to be sanitized.
        """
        # List all JSON files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                with open(file_path, "r") as file:
                    print(f"Sanitizing file {filename}")
                    try:
                        data = json.load(file)
                        # Check if "subtraction" field exists and needs sanitization
                        if "subtraction" in data and isinstance(data["subtraction"], List):
                            sanitized_subtraction = sanitize_subtraction_list(data["subtraction"])
                            data["subtraction"] = sanitized_subtraction
                            # Save the modified data back to the file
                            with open(file_path, "w") as outfile:
                                json.dump(data, outfile)
                    except json.JSONDecodeError:
                        print(f"Skipping file {filename} due to JSON decoding error.")

    def sanitize_subtraction_list(subtraction_list: List[str]) -> List[str]:
        """
        Converts a list of stringified boolean lists into a list of string booleans.
        Args:
        - subtraction_list (List[str]): The original list of stringified boolean lists.
        Returns:
        - List[str]: A sanitized list of string booleans.
        """
        sanitized_list = []
        # Regular expression to find all boolean values in the string
        boolean_pattern = re.compile(r"True|False", re.IGNORECASE)
        for item in subtraction_list:
            # Find all boolean values in the current string
            bool_strings = boolean_pattern.findall(item)
            # Convert boolean strings to their lowercase string representations and extend the sanitized list
            sanitized_list.extend([bs.lower() for bs in bool_strings])
        return sanitized_list

    def default_converter(o):
        try:
            # Attempt to use the object's own method to convert it to a serializable form, if available
            return o.__dict__
        except AttributeError:
            # If it doesn't work, just convert the object to a string
            return str(o)

    def crash_file_tracker(variable):
        crashfile = str(variable)
        with open(os.path.join(os.getcwd(), folder, "crash_file.txt"), "w") as tf:
            tf.write(crashfile)

    def store_and_merge(new_args, function_name):
        # Path to the JSON file
        file_path = os.path.join(os.getcwd(), folder, f"{function_name}.json")
        
        # Check if the file exists and read existing data; if not, initialize an empty dict
        if os.path.exists(file_path):
            with open(file_path, "r") as tf:
                try:
                    existing_data = json.load(tf)
                    # Ensure existing_data is a dictionary
                    if not isinstance(existing_data, dict):
                        existing_data = {}
                except json.JSONDecodeError:  # In case the file is empty or contains invalid JSON
                    existing_data = {}
        else:
            existing_data = {}
        
        # Merge the new arguments into the existing data
        for key, value in new_args.items():
            if key in existing_data:
                # Assume the value is a list and append the new value(s)
                if isinstance(value, list):
                    existing_data[key].extend(value)
                else:
                    # If the existing or new value is not a list, make it a list and append
                    if not isinstance(existing_data[key], list):
                        existing_data[key] = [existing_data[key]]
                    existing_data[key].append(value)
            else:
                # If the key does not exist in the existing data, add it
                existing_data[key] = value if isinstance(value, list) else [value]
        
        # Store the merged data
        with open(file_path, "w") as tf:
            json.dump(existing_data, tf)

    def normalize_booleans_in_string(json_str):
        # Find all occurrences of True/False and replace them with true/false
        if "True" in json_str or "False" in json_str:
            print("\033[91mFound and modifying booleans in JSON string.\033[0m")
        json_str = json_str.replace("True", "true").replace("False", "false")
        return json_str

    def serialize_proto_value(value):
        """Converts Protocol Buffer RepeatedComposite or other non-serializable values to Python native types."""
        if hasattr(value, 'extend'):  # Check if it behaves like a list
            return [v for v in value]
        else:
            return value

    def load_prompts(prompt_name):
        # Dynamically loading and printing all variables
        try:
            print("Loading Test-Prompts")
            with open('test_prompts.json', 'r') as json_file:
                data = json.load(json_file)
            return data.get(prompt_name)
        except:
            print("\033[91mError during loading of test prompts\033[0m")
            return None


    @dataclass
    class FunctionCall:
        name: Optional[str] = None
        arguments: Optional[str] = None  # Arguments as a JSON string or None

    @dataclass
    class ChatCompletionMessage:
        role: str
        function_call: Optional[FunctionCall] = None
        content: Optional[str] = None
        tool_calls: Optional[Any] = None


    # CREATE CODE AND TOOLS FROM LLM RESPONSE
    def create_code(PythonCode):
        # Preprocess the PythonCode to indent it correctly
        indented_code = "    " + PythonCode.replace('\n', '\n    ')
        # Construct the wrapped code with the pre-indented Python code
        wrapped_code = "def temp_function():\n" + indented_code + "\n    return locals()"
        try:
            local_env = {}
            exec(wrapped_code, {}, local_env)  # Execute the wrapped function definition
            result = local_env['temp_function']()  # Call the newly defined function and get the return value
            # Assuming the result is a dictionary of values you want to return
            return str(result)
        except Exception as e:
            return f"the function failed and returned: {str(e)}"

    def create_tool_from_json(directory_path):
        function_declarations = []
        for function_file in glob.glob(os.path.join(directory_path, "*.json")):
            with open(function_file, "r") as file:
                function_definition = json.load(file)
                
                # Remove the 'default' keys from parameters if they exist
                for param in function_definition['parameters']['properties'].values():
                    if 'default' in param:
                        del param['default']
                
                function_declaration = FunctionDeclaration(
                    name=function_definition["name"],
                    description=function_definition["description"],
                    parameters=function_definition["parameters"]
                )
                function_declarations.append(function_declaration)
        
        return Tool(function_declarations=function_declarations)

    # CLICKING (NOT YET FUNCTIONING)
    def return_click_point():
        #Store Info
        
        click_file_path = os.path.join(os.getcwd(), folder, f"{click_filename}.txt")
        if os.path.exists(click_file_path):
            with open(click_file_path, 'r') as file:
                click_data_loaded = json.load(file)    

        point_coords = click_data_loaded['point']
        vector_coords = click_data_loaded['vector']
        Click_Point = rs.CreatePoint(round(point_coords['x'], 2), round(point_coords['y'],2 ), round( point_coords['z'],2 ))
        Click_Normal = rs.CreateVector(vector_coords['x'], vector_coords['y'], vector_coords['z'])
        Click_string = f"The user clicked at the Position: {Click_Point[0]},{Click_Point[1]},{Click_Point[2]} " \
            f"with the normal facing in the direction: {Click_Normal[0]},{Click_Normal[1]},{Click_Normal[2]}"
        return Click_string


    # FUNCTIONS DEPENDENT ON LLM PROVIDER
    def toChatGPT(message): # appears to only be used in toGemini
        if debug == True:
            print("message to ChatGPT: ")
            print(message)
        function_name = None
        args_json = None

        # Check if there's a function call in the message
        if hasattr(message.parts[0], 'function_call') and message.parts[0].function_call:
            function_name = message.parts[0].function_call.name
            # Serialize arguments using the custom serializer
            args_dict = {key: serialize_proto_value(value) for key, value in message.parts[0].function_call.args.items()}
            args_json = json.dumps(args_dict, ensure_ascii=False, default=custom_serializer)

        # Safely check if there's text in the message
        message_text = None
        try:
            if message.parts[0].text:  # This line may raise ValueError if text is not present
                message_text = message.parts[0].text
        except AttributeError:
            pass  # No text in the message part, handle as needed

        # Create the FunctionCall object if a function call exists
        function_call = FunctionCall(name=function_name, arguments=args_json) if function_name and args_json else None

        # Create the ChatCompletionMessage object
        chat_message = ChatCompletionMessage(role='model', function_call=function_call, content=message_text)

        return chat_message

    def toGemini(full_message): # not in use
        # Extract the last message
        last_message = full_message[-1]
        
        # Determine the role and convert it to Gemini API role
        gemini_role = 'user' if last_message['role'] == 'user' else 'model'

        if gemini_role == 'user':
            return last_message['content']
        else:
            return ""

    def RequestFunction(assistant_message):
        '''
        Get function for function implementation from unique LLM response assistant_message
        '''
        if GPT == "Anthropic":
            class FunctionExtractor:
                def __init__(self, assistant_message):
                    # Extracting the ToolUseBlock from the assistant message
                    
                    try:
                        self._name = assistant_message[1].name
                        self._arguments = assistant_message[1].input
                        self._id = assistant_message[1].id
                    except:
                        print("Error During Reading of Function Call")
                        self._name = ''
                        self._arguments = {}
                        self._id = ''
                        

                @property
                def name(self):
                    return self._name

                @property
                def arguments(self):
                    return self._arguments
                
                @property
                def id(self):
                    return self._id

            # Creating and returning an instance of FunctionExtractor with the provided assistant message
            return FunctionExtractor(assistant_message)

        if GPT == "Mistral":
            return assistant_message.tool_calls[0].function

        if GPT == "Gemini":
            return assistant_message.function_call

        if GPT == "ChatGPT":      
            return assistant_message.function_call
        
    def append_function(assistant_message, answer = "", new_args = ""):
        '''
        Appends function implementation to messages
        '''
        returned_function_call = RequestFunction(assistant_message)
        function_name = returned_function_call.name


        if answer == "":
            content_message = "successfully created " + function_name + " object. Call now the next function!"
            print("generated answer: " + content_message)
        else:
            print("answered with: " + answer)
            content_message = answer

        if GPT == "Mistral":
            messages.append(assistant_message)
            messages.append(ChatMessage(role="tool", name=str(function_name), content=content_message))
            return messages
        
        elif GPT == "Gemini":
            messages.append({"role":"user","content": "Execute now the next function call. If you already executed all functions, call the finished function"})
            return messages
        
        elif GPT == "ChatGPT":
            messages.append({"role": "assistant", "function_call": {"name": str(function_name), "arguments": str(new_args)}})
            messages.append({"role": "user","content": content_message})
            #messages.append({"role":"user","content": "Execute now the next function call. If you already executed all functions, call the finished function"})
            return messages
        
        
        
        elif GPT == "Anthropic":
            messages.append({"role": "assistant", "content": assistant_message})
            id = returned_function_call.id

            messages.append({"role":"user","content": [{"type": "tool_result", "tool_use_id": id, "content": content_message}, {"type": "text", "text": "Please call the next function! If you are finished, call the finished function"}]})
            return messages

        else:
            print("Error, no GPT defined")

    def RequestText(assistant_message):
        '''
        Get function for content from unique LLM response assistant_message
        '''
        if GPT == "Mistral":
            try:
                return assistant_message.content
            except:
                print("Error During Reading of Answer")
                return None

        if GPT == "Gemini":
            try:
                return assistant_message.content
            except:
                print("Error During Reading of Answer")
                return None

        if GPT == "ChatGPT":      
            try:
                return assistant_message.content
            except:
                print("Error During Reading of Answer")
                return None
        
        if GPT == "Anthropic":
            try:
                return assistant_message[0].text
            except:
                print("Error During Reading of Answer")
                return None

    # REQUESTGPT
    # Define functions used to interact with each LLM, depends on the provider. 
    # Note that here, unlike with other function definitions, the "if" condition is placed _before_
    # the definition rather than having it inside
    if GPT == "ChatGPT":
        crash_file_tracker(messages)

        def RequestGPT(messages, tool_call = True, enforce = False):
            response = openai.chat.completions.create(
                model= OpenAI_model, # GPT 4: gpt-4-0613 GPT 4 Turbo: gpt-4-0125-preview GPT-4o : gpt-4o-2024-05-13
                messages=messages,
                functions=functions,
                temperature=0.2
            )
            crash_file_tracker(response)
            if debug == True:
                print("ChatGPT Answer: ")
                print(response.choices[0].message)
            return response.choices[0].message
            
    if GPT == "Anthropic":
        def RequestGPT(messages, tool_call = True, enforce = False):
            if tool_call:
                response = client.beta.tools.messages.create(
                    model=model, 
                    max_tokens=4092,
                    messages = messages,
                    tools=functions
                )
                print("Anthropic Answer: ")
                print(response.content)
                return response.content
            if tool_call == False: #Mal schauen, ob das so funktioniert, oder ob dann immer ein Tool-Call gemacht wird
                response = client.beta.tools.messages.create(
                    model=model, 
                    max_tokens=4092,
                    messages = messages,
                    tools=functions
                )
                print("Anthropic Answer: ")
                print(response.content)
                return response.content
            else:
                response = client.messages.create(
                    max_tokens=1024,
                    messages = messages,
                    model=model,
                )
                print("Anthropic Answer: ")
                print(response.content[0])
                return response.content[0]
                
    if GPT == "Mistral":

        def RequestGPT(messages, tool_call = True, enforce = False):
            crash_file_tracker(messages)
            client = MistralClient(api_key=api_key)
            if tool_call:
                if enforce:
                    tool_choice = "any"
                else:
                    tool_choice = "auto"
                response = client.chat(
                    model=model,
                    messages=messages,
                    tools=functions,
                    tool_choice=tool_choice)
            else:
                response = client.chat(
                    model=model,
                    tools=functions,
                    tool_choice="none",
                    messages=messages)
                
            if debug == True:
                print("Mistral Answer: ")
                print(response.choices[0].message)
            return response.choices[0].message

    if GPT == "Gemini":
        def RequestGPT(messages, tool_call = True, enforce = False):
            #Gemini WRAPPER:
            global Gemini_ChatHistory
            global response
            global last_message
            last_message = toGemini(messages)
            print("User Message: ")
            print(last_message)
            crash_file_tracker(last_message)


            if debug == True:
                print("Gemini Chat History: ")
                print(Gemini_ChatHistory)

            if tool_call:
                tool_config = ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
                ))

                model = GenerativeModel("gemini-1.5-pro-preview-0409",
                                        tools = [geometry_tool],
                                        tool_config=tool_config,
                                        generation_config={"temperature": 0.2})
            else:
                tool_config = ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
                ))            
                model = GenerativeModel("gemini-1.5-pro-preview-0409",
                                        tools = [geometry_tool],
                                        tool_config=tool_config,                                    
                                        generation_config={"temperature": 0.2})
                
            chat = model.start_chat(history = Gemini_ChatHistory)
            crash_file_tracker(str(Gemini_ChatHistory) + str(last_message))
            response = chat.send_message(last_message)

            Gemini_ChatHistory = chat.history

            print("Gemini Answer: ")
            crash_file_tracker(response.candidates[0].content)
            print(response.candidates[0].content)

            return toChatGPT(response.candidates[0].content)


    # -----------------------*** AGENT FUNCTIONS ***-----------------------
    def Agent1(user_input, instruction, messages, again):
        '''
        This agent enables the LLM to engage in dialogue with the user to solicit additional 
        information if it recognizes that the user prompts might lack clarity or completeness
        -Click functionality is not yet in operation
        -Loop allows the LLM to converse with user until it sends a TERMINATE message
        Args:
        -user_input:
        -messages: expandable log of conversation with LLM
        Return:
        -messages: updated log of conversation with LLM, includes clarifications
        '''

        global text
        #If click mode is activated, add user Click to Info:
        if click:
            click_message = return_click_point()
            complete_message = click_message + user_input
            text.append("[system]" + click_message)
            text.append("[User] " + user_input)
            messages.append({"role":"user","content": complete_message})
            print(complete_message)
        else:
            text.append("[User] " + user_input)
            messages.append({"role":"user","content": user_input})

        # loop to check and ask back
        while True:
            assistant_message = RequestGPT(messages, False)
            # exit if no content
            TextAnswer = RequestText(assistant_message)

            if TextAnswer:
                messages.append({"role":"assistant","content": TextAnswer})
                text.append("[GPT]" + TextAnswer)        

                # terminate conversation
                if "TERMINATE" in TextAnswer:
                    return messages
            
                # open window with question for user
                user_input = rs.EditBox(
                default_string = "",
                message = assistant_message.content,
                title = "RhinoGPT")
                user_input = input("assistant_message.content")
            
            else:
                return messages
            
            messages.append({"role":"user","content": user_input})
            text.append("[User] " + user_input)

        return messages

    def Agent2(instruction, messages, completion):
        '''
        This agent instructs the LLM to first generate step-by-step instructions 
        for geometry creation.
        Args:
        -instruction: description of the agent step (generating description of functions the LLM will call)
        -messages: expandable log of conversation with LLM
        Returns:
        -messages: updated log of conversation with LLM, includes description of function calls the LLM will call
        '''
        # Append system message as user Instruction
        user_input = instruction
        messages.append({"role":"user","content": user_input})
        text.append("[User] " + user_input)

        assistant_message = RequestGPT(messages, False)

        TextAnswer = RequestText(assistant_message)

        if TextAnswer:
            response = TextAnswer
            text.append("[GPT] " + str(response))
            messages.append({"role":"assistant","content": response})

        return messages

    def Anthropic_Agent3(instruction, messages, completion):

        user_input = instruction
        messages.append({"role": "user", "content": user_input})
        
        if FinishedCalled:
            print("Anthropic Finished already everything in Agent 2! Exiting")
            messages.append({"role":"user","content": "Anthropic Finished already everything in Agent 2! Exiting"})
            return messages

        function_answer = RequestGPT(messages, tool_call=True, enforce=True)

        counter = 0

        while counter < 15:
            counter += 1
            print("--------------------- Function Call Found ---------------------")
            print({"role": "assistant", "content": function_answer})
            messages.append({"role": "assistant","stop_reason": "tool_use", "content": function_answer})
            if FinishedCalled:
                break
            messages.append({"role":"user","content": "Please call the next Function! If you are finished, call the finished function"})
            function_answer = RequestGPT(messages, tool_call=True, enforce=True)

        completion = True
        return messages                

    def Agent3(instruction, messages, completion):
        '''
        Parses through the messages log, requests the creation of functions when it identifies a function message,
        identifies the function type (finished, createCode, or other) ***does something else I need to properly identify***
        and appends the working function to the message log.
        Args:
        -instruction: description of the agent step (generating description of functions the LLM will call)
        -messages: expandable log of conversation with LLM
        -completion: boolean indicating that stepPlanning step has already taken place; unsure of benefit here
        Returns:
        -messages: updated log of conversation with LLM, includes description of function calls the LLM will call
        ''' 

        # Append system message as user Instruction
        user_input = instruction
        messages.append({"role":"user","content": user_input})

        assistant_message = RequestGPT(messages, tool_call = True, enforce = True)
        crash_file_tracker(assistant_message)        
        # while function call (max Calls = 10)
        counter = 0

        returned_function_call = RequestFunction(assistant_message)

        while counter < 15: #This limit of 15 should be improved and be made case dependent
            counter += 1
            if returned_function_call == None or returned_function_call.name == '':
                print("no function call was appended!")
                response = assistant_message
                messages.append({"role":"assistant","content": response})
                messages.append({"role":"user","content": "Please call the next function! If you are finished, call the finished function"})
                assistant_message = RequestGPT(messages, tool_call = True, enforce = True)
                returned_function_call = RequestFunction(assistant_message)

            function_name = returned_function_call.name
            function_str = returned_function_call.arguments

            try: new_args = json.loads(function_str)
            except: 
                try: new_args = json_repair.loads(function_str)
                except:
                    try: 
                        new_args = json_repair.loads(repair_python_to_JSON(function_str))
                        print("final repair attempt successful!")
                    except: new_args = function_str
            
            # Ensure all lists in the arguments have the same length:
            new_args = parse_list(new_args)  #Should only be necessary for ChatGPT 4 and 4o, and not Turbo
            print(new_args)


            print("--------------------- Function Call Found ---------------------")
            print("Function name: ")
            print(function_name)

            if str(function_name) == "finished":
                messages.append({"role": "assistant", "function_call": {"name": str(function_name), "arguments": str(new_args)}})
                break

            elif str(function_name) == "CreateCode":
                print("---------------------- Execute Python Code ----------------------")
                arguments_dict = new_args
                python_code = arguments_dict["PythonCode"]
                try: function_response = create_code(str(python_code))
                except: function_response = "Unknown Execution Error during Code Creation"
                answer = "```Python_Code output: " + function_response + "```"
                print(answer)
                append_function(assistant_message, answer, new_args)
                print("Feeding Answer back to ", GPT,"...")
                assistant_message = RequestGPT(messages, tool_call = True, enforce = True)
                returned_function_call = RequestFunction(assistant_message)

            else:
                # get function name and arguments
                print("calling function:")
                
                crash_file_tracker(returned_function_call.arguments)

                # Path to the JSON file
                file_path = os.path.join(os.getcwd(), folder, f"{function_name}.json")

                # Check if the file exists and read existing data; if not, initialize an empty dict
                if os.path.exists(file_path):
                    with open(file_path, "r") as tf:
                        try:
                            existing_data = json.load(tf)
                            # Ensure existing_data is a dictionary
                            if not isinstance(existing_data, dict):
                                existing_data = {}
                        except json.JSONDecodeError:  # In case the file is empty or contains invalid JSON
                            existing_data = {}
                else:
                    existing_data = {}
                
                # Merge the new arguments into the existing data
                for key, value in new_args.items():
                    if key in existing_data:
                        # Assume the value is a list and append the new value(s)
                        if isinstance(value, list):
                            existing_data[key].extend(value)
                        else:
                            # If the existing or new value is not a list, make it a list and append
                            if not isinstance(existing_data[key], list):
                                existing_data[key] = [existing_data[key]]
                            existing_data[key].append(value)
                    else:
                        # If the key does not exist in the existing data, add it
                        existing_data[key] = value if isinstance(value, list) else [value]
                
                # Store the merged data
                with open(file_path, "w") as tf:
                    json.dump(existing_data, tf)

                # Add function call to messages

                append_function(assistant_message, "", new_args)

                logger(messages, logfile)
                print({"role": "assistant", "function_call": {"name": str(function_name), "arguments": str(new_args)}})
                    
                print("Calling for next Function")
                # Call again (assuming RequestGPT() is defined elsewhere and manages the loop)
                assistant_message = RequestGPT(messages, tool_call = True, enforce = True)
                returned_function_call = RequestFunction(assistant_message)                                                                                                                           

        print("Called all Functions")

        # if content -> output to user
        try:
            if assistant_message.content:
                response = RequestText(assistant_message)
                if completion:
                    global text
                    text.append("[GPT] " + response)
                messages.append({"role":"assistant", "content": response})

        except:
            print("No Content in Assistant Message")

        return messages


    # -----------------------*** MAIN PROGRAM ***-----------------------

    if go:
        
        # -----------------------*** START CONTACT WITH LLM API ***-----------------------

        messages = [{"role": "system", "content": "LOG RESET"}]
        text = []
        text.append("[Prog] New Conversation. Press \"Start\" to begin.")

        # reset function arguments
        arg_files = glob.glob("temp\*.json")

        for arg_file in arg_files:
            try:
                os.remove(arg_file)
            except OSError as e:
                print(f"Error deleting function argument files")

        with open(logfile, "w") as tf:
            json.dump(messages, tf)

        # Load and hand over API key, load function definitions and modify them to structure expected by each provider
        if GPT == "ChatGPT":
            key = open("key.txt", "r")
            api_key = key.read()
            openai.api_key = api_key

            functions = []
            # get provided functions, already in format expected by OpenAI
            for function_name in glob.glob("functions\*.json"):
                with open(os.path.join(os.getcwd(), function_name), "r") as function:
                    functions.append(json.load(function))

        if GPT == "Anthropic":
            key = open("anthropic_key.txt", "r")
            api_key = key.read()
            client = Anthropic(
                # This is the default and can be omitted
                api_key=api_key,
            )
            model = "claude-3-opus-20240229"  # or Expensive: "claude-3-opus-20240229" Fast: "claude-3-haiku-20240307" Intermediate: "claude-3-sonnet-20240229"
            
            functions = []
            # get provided functions, adapt them to format expected by Anthropic
            for function_name in glob.glob("functions/*.json"):
                with open(os.path.join(os.getcwd(), function_name), "r") as function_file:
                    function_data = json.load(function_file)
                    # Rename 'parameters' key to 'input_schema'
                    if 'parameters' in function_data:
                        function_data['input_schema'] = function_data.pop('parameters')
                    functions.append(function_data)

        if GPT == "Mistral":
            key = open("mistral_key.txt", "r")
            api_key = key.read()
            model = "mistral-large-2402"

            #Function calls are exactly the same as with ChatGPT
            functions = []
            # get provided functions
            for function_name in glob.glob("functions/*.json"): 
                with open(os.path.join(os.getcwd(), function_name), "r") as function_file:
                    # Load the original function data
                    function_data = json.load(function_file)

                    # Modify the structure as requested
                    modified_function = {
                        "type": "function",
                        "function": function_data
                    }
                    functions.append(modified_function)

        if GPT == "Gemini":
            geometry_tool = create_tool_from_json(directory_path)
            Gemini_ChatHistory = []

        # -----------------------*** DEFINE SYSTEM AND ASK FOR INITIAL USER INPUT ***-----------------------

        # taskfile contains the system and agent workflow descriptions, not the prompt!
        with open(taskfile, "r") as task_file:
            task = json.load(task_file)
        
        with open(logfile, "r") as log_file:
            messages = json.load(log_file)

        # "content" can be changed
        messages = [{"role": "system", "content": "LOG RESET"}]
        text = []
        text.append("[Prog] New Conversation. Press \"Start\" to begin.")
        status = "Programm has Started"

        # Find the dictionary with role 'system'
        for item in messages:
            if item["role"] == "system":
                # Update the content
                if item["content"] == "LOG RESET":
                    messages = []
                    again = False
                    ask = "Describe your request"
                else:
                    again = True
                    ask = "What do you want to add or change?"

        user_input = load_prompts(prompt_name)
        # get user input: (works on Rhino/GH interface)
        # user_input = rs.EditBox(
        #     default_string = "",
        #     message = ask,
        #     title = "RhinoGPT")

        system_message = str(task["system_message"])
        
        if GPT == "ChatGPT":
            messages.append({"role":"system","content": system_message})
        if GPT == "Mistral":
            messages.append({"role":"system","content": system_message})
        if GPT == "Anthropic":
            user_input = "system_message" + " User Input: " + user_input
        if GPT == "Gemini":
            user_input = "system_message" + " User Input: " + user_input


        # -----------------------*** AGENT WORFKLOW PROCESS ***-----------------------
        # Step-by-step process, in which each steps 1, 2, 4 represents an agent workflow:
        #   step1: askBack
        #   step2: stepPlanning
        #   step3: execute prepared function calls and make use of the createCode function if needed
        #   step4: visualInspection, not included in this script
        # The steps within each step are carried out via the Agent1, Agent2, and Agent3 functions.

        status = "Getting Instructions"
        instruction = str(task["step1"])
        
        if Allow_askback:
            status = "Starting Step1"
            # askBack
            logger(messages, logfile)
            messages = Agent1(user_input, instruction, messages, again)
            status = "Finished Step 1, starting Step 2"
            # Define next instruction
            instruction = str(task["step2"])  
        else:
            status = "Skipping Step 1"
            instruction = user_input + " | " + str(task["step2"])

        completion = False

        # stepPlanning
        logger(messages, logfile)
        messages = Agent2(instruction, messages, completion)
        status = "Finished Step 2, running Step 3"
        
        # Define next instruction
        logger(messages, logfile)
        instruction = str(task["step3"])
        completion = True

        # Execution
        logger(messages, logfile)
        messages = Agent3(instruction, messages, completion)
        status = "Finished Step 3, program run was succesfully"
        logger(messages, logfile)

        change = True

        if GPT == "Anthropic":
            print("attempting Sanitization of Subtraction Field")
            try: sanitize_subtraction_field(folder)
            except: print("Sanitization Failed")
            
        destination = copy_files_and_folders(Path(base_destination_path), new_folder_name)
        with open("destination.txt", 'w') as file:
            file.write(destination)

    elif reset:

        # -----------------------*** CLEAR LOG AND CONVERSATION ***-----------------------
        # Used only if go is false and reset is true

        #Clear stdout for logging:
        #sys.stdout = sys.__stdout__

        messages = [{"role": "system", "content": "LOG RESET"}]
        text = []
        text.append("[Prog] New Conversation. Press \"Start\" to begin.")

        # reset function arguments
        arg_files = glob.glob("temp\*.json")

        for arg_file in arg_files:
            try:
                os.remove(arg_file)
            except OSError as e:
                print(f"Error deleting function argument files")

        with open(logfile, "w") as tf:
            json.dump(messages, tf)