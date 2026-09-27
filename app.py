from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline



if __name__ == "__main__":
    docs = load_all_documents("data")
    pipeline = EmbeddingPipeline()
    chunks = pipeline.chunk_documents(docs)
    chunksvectors = pipeline.embed_chunks(chunks)
    print(chunksvectors)