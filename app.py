from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore



if __name__ == "__main__":
    # docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    store.load()
    results = store.query("what is real friendship?", top_k=3)
    print(results[0]['metadata'])
    