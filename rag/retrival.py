from rag.vector_db import get_embedding
from rag.vector_db import index_name

def retrieve_chunks(query,namespace):
    query_embedding = get_embedding(query)

    results = index_name.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        namespace=namespace
    )
    
    chunks = []
    for match in results.matches:
        chunks.append(match.metadata["text"])
    return chunks