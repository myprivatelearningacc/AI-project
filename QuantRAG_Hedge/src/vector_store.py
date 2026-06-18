from typing import List, Dict
import json
from pathlib import Path
import numpy as np


class VectorStore:
    """
    A simple local vector store using numpy arrays.
    """

    def __init__(self):
        self.embeddings = None
        self.chunks = None

    def build(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Store chunks and their embeddings.
        """

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        self.chunks = chunks
        self.embeddings = embeddings

    def save(self, save_dir: str):
        """
        Save embeddings and chunk metadata locally.
        """

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        np.save(save_path / "embeddings.npy", self.embeddings)

        with open(save_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(self, save_dir: str):
        """
        Load embeddings and chunk metadata from disk.
        """

        save_path = Path(save_dir)

        embeddings_path = save_path / "embeddings.npy"
        chunks_path = save_path / "chunks.json"

        if not embeddings_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                "Vector store files not found. Please build the vector store first."
            )

        self.embeddings = np.load(embeddings_path)

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)