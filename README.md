# RAG (Advanced and Vectorless)

## Why use an `EmbeddingManager` instead of a simple script?

While generating embeddings with a large language model (LLM) or a lightweight embedding model can theoretically be done in just a few lines of code (e.g., using `SentenceTransformer`), we opted for a more robust, object-oriented approach by creating an `EmbeddingManager` class. 

Here are the key reasons for this architectural decision:

1. **Load Model Once:** It ensures the model is loaded into memory only once during initialization, rather than reloading it every time an embedding is needed.
2. **Generate Embeddings:** It provides a structured, reusable method for generating embeddings for single text inputs.
3. **Batch Processing:** It can easily be extended to generate embeddings for multiple documents efficiently in batches.
4. **Calculate Dimensions:** It automatically checks and validates the embedding dimensions, ensuring compatibility with downstream tasks (like vector databases or similarity search).
5. **Handle Errors:** It incorporates `try-except` blocks to gracefully handle potential errors during model loading and inference, preventing silent failures.
6. **Flexibility (Change Models):** Using a class makes it trivial to swap out the underlying embedding model (e.g., switching from `all-MiniLM-L6-v2` to a different HuggingFace model) by simply passing a new `model_name` parameter.
7. **Future-Proofing (Caching):** This structure lays the groundwork for adding advanced features later, such as embedding caching, to avoid recomputing vectors for previously processed text.
