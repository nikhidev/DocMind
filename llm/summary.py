from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os 
load_dotenv()
my_token = os.getenv("my_token")

client = InferenceClient(
    api_key=my_token
)
def summarize_text(text):

    prompt = f"""
    Analyze the document and provide.
    Provide :
    1.Executive summary
    2.Key Points  
    3.Important Insights
    4.Action Items(if any)

    paper : {text[:8000]}
    """
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content