# embedding_retrieval.py (Hugging Face version)
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
import os

# Initialize HF model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma DB
chroma_client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))
collection = chroma_client.get_or_create_collection(name="embeddings")


def query_embeddings(query_text: str, top_k: int = 5):
    """Query Chroma DB using HF embeddings and return relevant contents"""
    # 1️⃣ Generate query embedding
    query_vector = model.encode(query_text).tolist()

    # 2️⃣ Search in Chroma
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    # 3️⃣ Extract content
    contents = [item["content"] for item in results["metadatas"][0]]
    return contents


# Example usage
if __name__ == "__main__":
    query = "Summarize my last trade note"
    results = query_embeddings(query)
    print("Top results:")
    for r in results:
        print("-", r)
