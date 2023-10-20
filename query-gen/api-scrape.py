# Code used to generate data for files "openai_.." A json from parsing the website. 
# The list of urls is imported from *_url_list.py

import os, requests, markdown, openai, json, concurrent.futures, threading
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime
from selenium import webdriver 
from selenium.webdriver.firefox.options import Options 

load_dotenv()

OPEANAI_MAX_TOKENS = 8192
ai_agent = "openai"  # Change this to "anthropicai" to use Claude-v1 and "openai" to use GPT-4-0314.
api_key = os.environ.get('OPENAI_KEY')


def get_ai_response(pre_prompt, prompt, ai_agent):
    openai.api_key = api_key

    prompt = prompt.replace("<p>", "")
    prompt = prompt.replace("</p>", "")

    prompt = prompt.replace("<pre>", "")
    prompt = prompt.replace("</pre>", "")
    prompt = prompt.replace("<code>", "")
    prompt = prompt.replace("</code>", "")
    prompt = prompt.replace("<blockquote>", "")
    prompt = prompt.replace("</blockquote>", "")
    while "  " in prompt:
        prompt = prompt.replace("  ", " ")
    while "\n\n" in prompt:
        prompt = prompt.replace("\n\n", "\n")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": pre_prompt},
                {"role": "user", "content": prompt},
            ],
            # max_tokens=(OPEANAI_MAX_TOKENS - len(prompt) - len(pre_prompt)),
            n=1,
            stop=None,
            temperature=0.5,
        )
        # print(response.choices[0].message["content"].strip())
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print("Failed prompt:", prompt)
        print(e)
        return None 
    

# openai.api_key = "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"
# api_key1 = "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"
# api_key2 = "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"
# api_key3 = "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"
# TODO: this is the prompt for torch hub
# pre_prompt = "We want to create a compact representation of the API. Convert the following documentation into \
#     a json with fields {domain, framework, functionality, api_name, api_call, api_arguments, \
#     python_environment_requirements, example_code, performance, description}. \
#     Make sure the `performance` has a subfield `dataset` corresponding to subfield `accuracy`. \
#     `domain` should include one of {Multimodal, Computer Vision, NLP, Audio, Tabular, Reinforcement Learning}"  

# # TODO: this is the prompt for huggingface
# pre_prompt = "We want to create a compact representation of the API. Convert the following documentation into \
#     a valid JSON line with fields {domain, framework, functionality, api_name, api_call, api_arguments, \
#     python_environment_requirements, example_code, performance, description}.\n \
#     The `api_name` should be the name of the model id, and the `api_call` should be one line of python code calling the api. \
#     The `example_code` should be directly copied from the website if there are any examples.\
# `domain` should include one of {Multimodal Feature Extraction, Multimodal Text-to-Image, Multimodal Image-to-Text, Multimodal Text-to-Video, \
# Multimodal Visual Question Answering, Multimodal Document Question Answer, Multimodal Graph Machine Learning, Computer Vision Depth Estimation,\
# Computer Vision Image Classification, Computer Vision Object Detection, Computer Vision Image Segmentation, Computer Vision Image-to-Image, \
# Computer Vision Unconditional Image Generation, Computer Vision Video Classification, Computer Vision Zero-Shor Image Classification, \
# Natural Language Processing Text Classification, Natural Language Processing Token Classification, Natural Language Processing Table Question Answering, \
# Natural Language Processing Question Answering, Natural Language Processing Zero-Shot Classification, Natural Language Processing Translation, \
# Natural Language Processing Summarization, Natural Language Processing Conversational, Natural Language Processing Text Generation, Natural Language Processing Fill-Mask,\
# Natural Language Processing Text2Text Generation, Natural Language Processing Sentence Similarity, Audio Text-to-Speech, Audio Automatic Speech Recognition, \
# Audio Audio-to-Audio, Audio Audio Classification, Audio Voice Activity Detection, Tabular Tabular Classification, Tabular Tabular Regression, \
# Reinforcement Learning Reinforcement Learning, Reinforcement Learning Robotics }\
#     Make sure the `performance` has a subfield `dataset` corresponding to subfield `accuracy`."  

# '''
# `domain` should include one of {Multimodal Feature Extraction, Multimodal Text-to-Image, Multimodal Image-to-Text, Multimodal Text-to-Video, \
# Multimodal Visual Question Answering, Multimodal Document Question Answer, Multimodal Graph Machine Learning, Computer Vision Depth Estimation,\
# Computer Vision Image Classification, Computer Vision Object Detection, Computer Vision Image Segmentation, Computer Vision Image-to-Image, \
# Computer Vision Unconditional Image Generation, Computer Vision Video Classification, Computer Vision Zero-Shor Image Classification, \
# Natural Language Processing Text Classification, Natural Language Processing Token Classification, Natural Language Processing Table Question Answering, \
# Natural Language Processing Question Answering, Natural Language Processing Zero-Shot Classification, Natural Language Processing Translation, \
# Natural Language Processing Summarization, Natural Language Processing Conversational, Natural Language Processing Text Generation, Natural Language Processing Fill-Mask,\
# Natural Language Processing Text2Text Generation, Natural Language Processing Sentence Similarity, Audio Text-to-Speech, Audio Automatic Speech Recognition, \
# Audio Audio-to-Audio, Audio Audio Classification, Audio Voice Activity Detection, Tabular Tabular Classification, Tabular Tabular Regression, \
# Reinforcement Learning Reinforcement Learning, Reinforcement Learning Robotics \
# }
# '''

# TODO: this is the prompt for Tensorflow Hub
pre_prompt = "We want to create a compact representation of the API. Convert the following documentation into \
    a valid JSON line with fields {domain, framework, functionality, api_name, api_call, api_arguments, \
    python_environment_requirements, example_code, performance, description}.\n \
    The `api_name` should be the name of the model id, and the `api_call` should be one line of python code calling the api. \
    The `example_code` should be directly copied from the website if there are any examples.\
    Make sure the `performance` has a subfield `dataset` corresponding to subfield `accuracy`.\
    Return only the valid json, and NO other text." 
write_lock = threading.Lock()


def process_url(args):
    """
    Args needs 3 things
    1. Identifier - short id specified by user
    2. URL - url
    3. Domain - what the API provided roughly does
    """
    identifier, url, domain = args[0], args[1], args[2]
    current_date = datetime.now().strftime('%y%m%d')
    try:
        # print(f"Processing URL: {url}")
        options = Options() 
        options.add_argument("-headless") 
        
        # using Firefox headless webdriver to secure connection to Firefox 
        with webdriver.Firefox(options=options) as driver: 
            # opening the target website in the browser 
            driver.get(url) 
 
            print("Page URL:", driver.current_url) 
            print("Page Title:", driver.title) 

            # Wait for page to load
            driver.implicitly_wait(10)

            # Get the page source after it has loaded
            page_source = driver.page_source

            soup = BeautifulSoup(page_source, 'html.parser')
            prompt = f"DOMAIN:{domain}, {markdown.markdown(soup.get_text())}"[:OPEANAI_MAX_TOKENS]
            result_text = get_ai_response(pre_prompt, prompt, ai_agent)
            file_name = f"{ai_agent}_{identifier}-{current_date}.json"
            with write_lock:
                with open(file_name, "a") as f:
                    if ai_agent == "openai":
                        result = json.loads(result_text)
                        result["domain"] = domain
                        f.write(json.dumps(result) + "\n")
                    # else:
                    #     f.write("{" + f"'domain':{domain}  {result_text} + '\n'")
    
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Failed URL: {url}")
        print(result)
        file_name = f"{ai_agent}_{identifier}-error-{current_date}.json"
        with write_lock:
            with open(file_name, "a") as f:
                f.write(url + "\n")
    

def scraper(urls):
    """
    For each url in urls, it needs to have id, url, and domain
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_url, url) for url in urls]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Exception: {e}")

def main():
    scraper([['oai',"https://platform.openai.com/docs/api-reference/introduction", 'generic']])

if __name__ == "__main__":
    main()