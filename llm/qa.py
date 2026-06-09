from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
load_dotenv()
my_token = os.getenv("my_token")

client = InferenceClient(
    api_key=my_token
)
def answer_question(question,context):

    prompt = f"""
    answer_question based on the context provided.

    Context : {context}
    Question : {question}
    Answer :
    """
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content