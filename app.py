import gradio as gr
import openai

import json
import yaml
import re
import os
from dotenv import load_dotenv

from utils import OpenAIClient

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

def api_to_inst_api_pair(file):
    openai.api_key = 'YOUR_API_KEY'

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100  # Adjust as needed.
    )

    generated_queries = response.choices[0].text.strip()
    

def execute(file):
    """
    Input:
    file: File - Swagger File containing API info.
    """
    # Validate that the uploaded file is a swagger file
    if not is_swagger_file(file):
        return "Invalid Swagger file. Please upload a valid Swagger file (JSON or YAML)."
    
    inst_api_pair = api_to_inst_api_pair(file)



def main():
    # Define Gradio interface with File input
    iface = gr.Interface(
        fn=predict,
        inputs=gr.inputs.File(label="Upload a File"),
        outputs="text",
    )

    # Launch the Gradio interface
    iface.launch()

if __name__ == "__main__":
    main()