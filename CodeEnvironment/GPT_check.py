# Overall, this code seems to be much more smoothly written than the ChatGPT one, but it does not make use of the functions established there

if allow_run:
    
    import base64
    import openai
    from openai import OpenAI
    from io import BytesIO
    from PIL import Image
    import os
    import json
    import rhinoscriptsyntax as rs

    os.chdir(path)
    change = False

    taskfile = "task.json"

    # get task
    with open(taskfile, "r") as tf:
        task = json.load(tf)

    # load API-Key
    key = open('key.txt', 'r')
    api_key = key.read()

    # define logfile
    logfile = "log.json"

    # hand over API-Key
    #OPENAI_API_KEY = api_key

    def encode_image(image_path, max_image=512):
        with Image.open(image_path) as img:
            width, height = img.size
            max_dim = max(width, height)
            if max_dim > max_image:
                scale_factor = max_image / max_dim
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height))

            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str

    key = open("key.txt", "r")
    api_key = key.read()
    openai.api_key = api_key

    #client = OpenAI()
    image_file = image
    max_size = 512  # set to maximum dimension to allow (512=1 tile, 2048=max)
    encoded_string = encode_image(image_file, max_size)

    system_prompt = task["step4"]
    user = ("Here is the image of the part. Does it look like requested?")

    with open(logfile, "r") as tf:
        messages_input = json.load(tf)

    # function calling need to be deleted, because it can not be processed with this GPT-version
    messages = [entry for entry in messages_input if "function_call" not in entry]

    # replace system message
    for item in messages:
        if item["role"] == "system":
        # Update the content
            item["content"] = system_prompt

    # insert user message and picture
    messages.append(
        {
            "role": "user",
            "content":
            [
                {"type": "text", "text": user},
                {
                    "type": "image_url",
                    "image_url": {"url":
                        f"data:image/jpeg;base64,{encoded_string}"}
                }
            ]
        }
    )

    if go:
        # apiresponse = client.chat.completions.with_raw_response.create(
        #     model="gpt-4-vision-preview",
        #     messages=messages,
        #     max_tokens=500,
        # )

        # currently set up only for ChatGPT
        apiresponse = openai.chat.completions.with_raw_response.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
        )
        debug_sent = apiresponse.http_request.content
        chat_completion = apiresponse.parse()
        print(chat_completion.choices[0].message.content)
        print(chat_completion.usage.model_dump())
        print(
            "remaining-requests: "
            f"{apiresponse.headers.get('x-ratelimit-remaining-requests')}"
        )

        str_message = rs.EditBox(
            default_string = "Describe the contents and layout of my image.",
            message = chat_completion.choices[0].message.content,
            title = "GPT answer"
        )

        response = chat_completion.choices[0].message.content

        if "CHANGE" in response:
            # add message
            response = response.replace("CHANGE","")
            messages_input.append({"role":"user","content":"Please check the result"})
            messages_input.append({"role":"assistant","content":response})
            # store in log file
            with open(logfile, "w") as tf:
                json.dump(messages_input, tf)
            # improve
            change = True