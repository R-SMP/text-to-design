# text-to-design
We present a novel framework leveraging function calling and agent workflows for automating Computer-Aided Design (CAD) geometry generation. This framework allows LLMs to automatically interpret intricate design requirements from textual prompts and generate the necessary CAD code for design realization. It can be integrated with any recent LLM that can be accessed via API. It offers four distinct agent workflows: zeroShot, stepPlanning, askBack, and visualInspection.

## RhinoGPT:
Main interface of program. After cloning repository, need to change paths in Base_destination and directory panels, as well as panel previous to geometry creation.

## CodeEnvironment:
**functions folder:** block.json, cylinder.json, finished.json, force.json, CreateCode.json

**temp folder:** instances of functions (JSON files), crash_file.txt, terminal_output.txt keys for ChatGP (key.txt), Anthropic, and Mistral

**geometry_creation.py:** module containing function defintions for functions that parse JSON object instances into rhinoscriptsyntax functions, imported into CreateGeometries python block in GH

**log.json:** contains log of conversation with LLM, including the system definition, prompt, and function instantiations

**task.json:** contains the descriptions of the system and each agent workflow (zeroShot, askBack, stepPlanning, visualInspection)

**test_prompts.json:** contains text prompts

### LLM provider key is to be placed in CodeEnvironment.
