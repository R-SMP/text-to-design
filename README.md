# text-to-design

We present a novel framework leveraging function calling and agent workflows for automating Computer-Aided Design (CAD) geometry generation. This framework allows LLMs to automatically interpret intricate design requirements from textual prompts and generate the necessary CAD code for design realization. It can be integrated with any recent LLM that can be accessed via API. It offers four distinct agent workflows: zeroShot, stepPlanning, askBack, and visualInspection.

## RhinoGPT:

Main interface of program. After cloning repository, need to change paths in Base\_destination and directory panels, as well as panel previous to geometry creation.

## RhinoGPT\_evaluation:

Start with installing all requirments and install the rhino\_Scripts\_requirments inside your local …/.rhinocode/py39-rh8/Scripts for working within Grasshopper/Rhino3D.



For Pytorch3d: https://github.com/facebookresearch/pytorch3d



The Evaluation of RhinoGPT:



Open RhinoGPT\_evaluation and start with changing all paths:



Panels:

* Base\_destination
* directory

Inside Python-Scripts:

* json-reader (.../CodeEvironmen/evaluation\_task.json)
* Create-Geometries (.../CodeEnvironment)
* Point-cloud-Generation (.../own\_benchmark)
* CLGD\_aglignment (add path to cloned GitHub repo of DDF-Loss: https://github.com/rsy6318/DDF-Loss)
* CLGD\_value (add path to cloned GitHub repo of DDF-Loss: https://github.com/rsy6318/DDF-Loss)
* STLtomesh (.../own\_benchmark)
* save (.../Evaluation)
* pic (.../Evaluation/Pictures)



After changing all paths, add all keys for API Access:

* OpenAI: key.txt
* Claude: anthropic.txt
* deepseek: deepseek\_key.txt
* Gemini: google\_proj\_id.txt



IMPORTANT for GEMINI: Add your json key file from your service-acc into local path on your computer:

* Inside GPT: os.environ\["GOOGLE\_APPLICATION\_CREDENTIALS"] = "(Change path to the json key file)"







**Process of (zeroShot,stepPlanning and askBack):**

* start: True (Once)

**LOOP:**

* IMPORTANT(Next): Press
* LLM (allow\_generation): True

wait

* EVALUATION (allow\_evaluation): True

wait

* SAVE: Press
* allow\_evaluation/allow\_generation: False

**start with loop again**







**Process of (visualInspection):**

* start: True (Once)

**LOOP:**

* IMPORTANT(Next): Press
* LLM (allow\_generation): True

wait

* visual\_check: Press

wait

* If Improve == True then:

 	visual\_improve : TRUE

 	wait



* EVALUATION (allow\_evaluation): True

wait

* SAVE: Press
* allow\_evaluation/allow\_generation/if needed also visual\_improve: False

**start with loop again**













## CodeEnvironment:

**functions folder:** block.json, cylinder.json, finished.json, force.json, CreateCode.json

**temp folder:** instances of functions (JSON files), crash\_file.txt, terminal\_output.txt keys for ChatGP (key.txt), Anthropic, and Mistral

**geometry\_creation.py:** module containing function defintions for functions that parse JSON object instances into rhinoscriptsyntax functions, imported into CreateGeometries python block in GH

**log.json:** contains log of conversation with LLM, including the system definition, prompt, and function instantiations

**task.json:** contains the descriptions of the system and each agent workflow (zeroShot, askBack, stepPlanning, visualInspection)

**test\_prompts.json:** contains text prompts

### LLM provider API key is to be stored inside CodeEnvironment.

