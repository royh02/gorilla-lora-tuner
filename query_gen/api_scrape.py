# Code used to generate data for files "openai_.." A json from parsing the website. 
# The list of urls is imported from *_url_list.py

import os, requests, markdown, openai, json, concurrent.futures, threading
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime
from selenium import webdriver 
from selenium.webdriver.firefox.options import Options 

load_dotenv()

DATA_FOLDER = os.environ.get("DATA_FOLDER")
ERROR_FOLDER = os.environ.get("ERROR_FOLDER")
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

    personal_data_dir = os.path.join(DATA_FOLDER, identifier)
    personal_error_dir = os.path.join(ERROR_FOLDER, identifier)

    os.makedirs(personal_data_dir, exist_ok=True)
    os.makedirs(personal_error_dir, exist_ok=True)

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
                file_loc = os.path.join(personal_data_dir, file_name)
                with open(file_loc, "a") as f:
                    if ai_agent == "openai":
                        result = json.loads(result_text)
                        result["domain"] = domain
                        print('asdfgh',result)
                        f.write(json.dumps(result) + "\n")
                    # else:
                    #     f.write("{" + f"'domain':{domain}  {result_text} + '\n'")
    
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Failed URL: {url}")
        file_name = f"{ai_agent}_{identifier}-error-{current_date}.json"
        file_loc = os.path.join(personal_error_dir, file_name)
        with write_lock:
            with open(file_loc, "a") as f:
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
    scraper([['boto3',"https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/acm-pca/client/get_policy.html", 'generic']])

if __name__ == "__main__":
    main()