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
            #user_input = rs.EditBox(
            #    default_string = "",
            #    message = assistant_message.content,
            #    title = "RhinoGPT")
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
    '''    

    # Append system message as user Instruction
    user_input = instruction
    messages.append({"role":"user","content": user_input})

    assistant_message = RequestGPT(messages, tool_call = True, enforce = True)
    crash_file_tracker(assistant_message)        
    # while function call (max Calls = 10)
    counter = 0

    returned_function_call = RequestFunction(assistant_message)

    while counter < 15:
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
