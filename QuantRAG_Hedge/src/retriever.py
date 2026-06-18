from typing import List, Dict
import numpy as np


class Retriever:
    """
    Retrieve top-k most relevant chunks based on cosine similarity.
    Since embeddings are normalized, dot product is cosine similarity.
    """

    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k chunks for a query.
        """

        if self.vector_store.embeddings is None:
            raise ValueError("Vector store is empty")

        query_embedding = self.embedder.embed_query(query)

        scores = np.dot(self.vector_store.embeddings, query_embedding)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            chunk = self.vector_store.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        return results