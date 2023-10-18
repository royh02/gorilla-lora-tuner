import os
import openai

model = "gpt-4"
system_message = "You are an AI assistant that generates instructions from Swagger API specs."

class OpenAIClient:
    def __init__(self, api_key, model=model, system_message=system_message):
        openai.api_key = api_key
        self.model = model
        self.system_message = system_message

    def chatgpt(self, query):
        messages = [
            {"role":"system", "content":self.system_message},
            {"role":"user", "content":query}
        ]
        response = openai.ChatCompletion.create(model=self.model, messages=messages).choices[0].message.content
        return response
    
    @staticmethod
    def get_embedding(query):
        query = query.replace("\n", " ")
        return openai.Embedding.create(input = [query], model="text-embedding-ada-002")['data'][0]['embedding']