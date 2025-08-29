# embedding_retrieval.py (Chroma new version, fixed include)

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# -------------------------------
# Initialize HF model
# -------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# Initialize Chroma DB (new syntax)
# -------------------------------
chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="embeddings")


def query_embeddings(query_text: str, top_k: int = 5):
    """
    Query Chroma DB using HF embeddings and return relevant contents.
    Compatible with ingestion using .mappings() and dictionary-based metadata.
    """
    query_vector = model.encode(query_text).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["metadatas", "embeddings"]  # removed 'ids'
    )

    contents = [metadata.get("content", "") for metadata in results["metadatas"][0]]
    return contents


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    query = "Summarize my last trade note"
    results = query_embeddings(query)
    print("Top results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}")
