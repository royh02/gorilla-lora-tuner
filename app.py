import gradio as gr
import openai

import json
import yaml
import re
import os
from dotenv import load_dotenv

from utils import OpenAIClient
from query_gen import 

load_dotenv()

OPENAI_KEY = os.environ.get('OPENAI_KEY')

def is_swagger_content(content):
    try:
        # Attempt to parse the content as JSON
        json.loads(content)
        return True
    except json.JSONDecodeError:
        try:
            # Attempt to parse the content as YAML
            yaml.safe_load(content)
            return True
        except yaml.YAMLError:
            return False

def is_swagger_file(file):
    if re.search(r"\.(json|yaml|yml)$", file.name, re.IGNORECASE):
        with open(file.name, 'rb') as f:
            content = f.read().decode('utf-8')
            return is_swagger_content(content)
    return False


    

def execute(inputs):
    """
    Input:
    inputs -> id, url, domain
    """
    
    progress=gr.Progress(track_tqdm=True)

    inst_api_pair = api_to_inst_api_pair(file)



def main():
    # Define Gradio interface with File input
    iface = gr.Interface(
        fn=execute,
        inputs=[
            gr.inputs.Textbox(label="ID"),
            gr.inputs.Textbox(label="URL"),
            gr.inputs.Textbox(label="Domain")
        ],
        outputs=gr.outputs.Textbox(label="Result"),
    )

    # Launch the Gradio interface
    iface.launch()

if __name__ == "__main__":
    main()