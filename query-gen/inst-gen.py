import openai
import json
import concurrent.futures
import threading
import os
import time

DATA_FOLDER = "/data/roy.huang/lora/data/api"
OUTPUT_FOLDER = "/data/roy.huang/lora/data/inst"
ERROR_FOLDER = "/data/roy.huang/lora/data/error"
PROMPT_FILE = "../data/prompts/pre-prompt.txt"
write_output_lock = threading.Lock()
write_error_lock = threading.Lock()
MAX_WORKERS = 2

PRE_PROMPT = open(PROMPT_FILE).read()

def call_chat_completion_api(data_entry, output_file_path, error_file_path):
    try:
        openai.API_KEY = os.environ.get("OPENAI_KEY")
        data_entry_string = json.dumps(data_entry)
        responses = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "system", "content": PRE_PROMPT},
                      {"role": "user", "content": data_entry_string}],
            n=1,
            temperature=0,
        )
        instruction = responses['choices'][0]['message']['content'].strip()
        data_entry['instruction'] = instruction

        with write_output_lock:
            with open(output_file_path, "a") as f:
                json.dump(data_entry, f)
                f.write("\n")
    except Exception as e:
        if "Rate limit reached" in str(e):
            print("Rate limit reached. Sleeping for 60 seconds.")
            time.sleep(65)  # Sleep for 60 seconds
            return call_chat_completion_api(data_entry, output_file_path, error_file_path)
        else:
            print(e)
            with write_error_lock:
                with open(error_file_path, "a") as f:
                    json.dump(data_entry, f)
                    f.write("\n")

def process_file(filename):
    full_input_path = os.path.join(DATA_FOLDER, filename)
    full_output_path = os.path.join(OUTPUT_FOLDER, filename)
    full_error_path = os.path.join(ERROR_FOLDER, filename)
    
    with open(full_input_path, "r") as f:
        data_list = json.load(f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(lambda data_entry: call_chat_completion_api(data_entry, full_output_path, full_error_path), data_list)


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(ERROR_FOLDER):
        os.makedirs(ERROR_FOLDER)

    filenames = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
    for _file in filenames:
        process_file(_file)