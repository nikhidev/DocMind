from pinecone import Pinecone
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.getenv("pinecone_key"))
index_name = pc.Index(os.getenv("pinecone_index_name"))

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    embedding = model.encode(text).tolist()
    return embedding

def store_chunks(chunks, namespace):
     vectors = []

     for i, chunk in enumerate(chunks):
         embedding = get_embedding(chunk)
         vectors.append({
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {"text": chunk}
            })
try:
        n
        result = index_name.upsert(vectors=vectors, namespace=namespace)
        print(result)
except Exception as e:
        print("Pinecone Error:", e)