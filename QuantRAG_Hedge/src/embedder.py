from typing import List
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class Embedder:
    """
    Convert text into vector representations without downloading external models.

    This uses HashingVectorizer, so it works offline and is enough for a basic RAG milestone.
    """

    def __init__(self, n_features: int = 2048):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm=None
        )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        """

        vectors = self.vectorizer.transform(texts)
        vectors = normalize(vectors, norm="l2", axis=1)

        return vectors.toarray()

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed one query.
        """

        vector = self.vectorizer.transform([query])
        vector = normalize(vector, norm="l2", axis=1)

        return vector.toarray()[0]