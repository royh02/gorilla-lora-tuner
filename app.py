from anyio import sleep
import gradio as gr

from query_gen import scraper, generate_inst
import uuid

from gorilla.alpaca_finetuning_v1.util import run_finetune
from gorilla.inference import infer

#Function for first part to generate the adaptor
def generate_adapter(ID, URL_List, Domain):
#Add your function here. Make sure that you return a file that is the adaptor
    url_lst = URL_List.split("\n")
    formatted_urls = []
    uid = f"{ID}_{str(uuid.uuid4())}" if ID else str(uuid.uuid4())

    for url in url_lst:
        formatted_urls.append([uid, url, Domain])

    scraper(formatted_urls)
    generate_inst(uid)

    gpus = min(len(url_lst), 8)

    adapter_file = run_finetune(uid, num_gpus=gpus, batch_size=1) # adjust as needed
    #This is where you return the file
    return uid, adapter_file

#Function to generate inference responses in a chatbot format
# def inference_response_chatbot(input, history):
# #Have the inference with the prompt input return text outpjut appropriately
#     return 

#Function to generate inference responses in a single text format
def inference_response_chatbot(uid, query):
#Have the inference with the prompt input return text outpjut appropriately
    return infer(query, uid)



def main():
    #This is the interface used for generating the adaptor
    finetuning = gr.Interface(
        fn=generate_adapter,
        inputs=["text", "text", "text"],
        outputs=[gr.Textbox(label="Assigned ID (Used for Inference)"), gr.File(label="Adapter", elem_id="result-adapter")],
        allow_flagging="never",

    )

    #interface for playground. Have included two options, a single text field for input and for output or a more chat gpt like plaground
    # playground_chatbot = gr.ChatInterface(inference_response_chatbot)

    playground_single_text = gr.Interface(
        fn=inference_response_chatbot,
        inputs=[
            gr.Textbox(label="ID"),
            gr.Textbox(label="Input")],
        outputs=["text"],
        allow_flagging="never",
    )
    
    demo = gr.TabbedInterface([finetuning, playground_single_text], ["Finetuning", "Playground"])
    demo.queue(max_size=10)
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()

