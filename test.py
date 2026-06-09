from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
load_dotenv()
my_token = os.getenv("my_token")

client = InferenceClient(
    api_key=my_token
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role":"user","content":"Explain the concept of machine learning in simple terms."}]
)
print(response.choices[0].message.content)
