from pinecone import Pinecone
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.getenv("pinecone_key"))
index = pc.Index(os.getenv("pinecone_index_name"))

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text).tolist()

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
        result = index.upsert(
            vectors=vectors,
            namespace=namespace
        )

        print("Upsert Result:", result)
        print(f"{len(vectors)} chunks stored in Pinecone!")

    except Exception as e:
        print("Pinecone Error:", e)
