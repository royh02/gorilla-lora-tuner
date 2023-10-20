from anyio import sleep
import gradio as gr

#Function for first part to generate the adaptor
def generate_adapter(ID, URL_List, Domain):
#Add your function here. Make sure that you return a file that is the adaptor
    for i in range(10):
        print(1)
    #This is where you return the file
    return 

#Function to generate inference responses in a chatbot format
def inference_response_chatbot(input, history):
#Have the inference with the prompt input return text outpjut appropriately
    return 

#Function to generate inference responses in a single text format
def inference_response_chatbot(input):
#Have the inference with the prompt input return text outpjut appropriately
    return 



def main():
    #This is the interface used for generating the adaptor
    finetuning = gr.Interface(
        fn=generate_adapter,
        inputs=["text", "text", "text"],
        outputs=gr.File(label="Adapter", elem_id="result-adapter"),
        allow_flagging="never",

    )

    #interface for playground. Have included two options, a single text field for input and for output or a more chat gpt like plaground
    playground_chatbot = gr.ChatInterface(inference_response_chatbot)

    playground_single_text = gr.Interface(
        fn=inference_response_chatbot,
        inputs=["text"],
        outputs=["text"],
        allow_flagging="never",

    )

    
    
    demo = gr.TabbedInterface([finetuning, playground_chatbot], ["Finetuning", "Playground"])

    demo.launch()

if __name__ == "__main__":
    main()

