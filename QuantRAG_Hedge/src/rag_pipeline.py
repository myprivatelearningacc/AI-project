from pathlib import Path
from typing import Dict, List, Tuple

from src.document_loader import load_markdown_docs
from src.chunker import chunk_documents
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.prompt_builder import build_prompt
from src.generator import Generator


class BasicRAGPipeline:
    """
    End-to-end Basic RAG pipeline.

    This class is used by the Streamlit app:
    - load documents
    - chunk documents
    - embed chunks
    - save/load vector store
    - retrieve top-k chunks
    - generate answer
    """

    def __init__(
        self,
        raw_docs_dir: str,
        vector_store_dir: str,
        chunk_size: int = 700,
        overlap: int = 120,
    ):
        self.raw_docs_dir = Path(raw_docs_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.generator = Generator()

    def vector_store_exists(self) -> bool:
        embeddings_path = self.vector_store_dir / "embeddings.npy"
        chunks_path = self.vector_store_dir / "chunks.json"
        return embeddings_path.exists() and chunks_path.exists()

    def build_index(self) -> Dict:
        """
        Build vector store from markdown documents.
        """

        docs = load_markdown_docs(str(self.raw_docs_dir))
        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        self.vector_store.build(chunks, embeddings)
        self.vector_store.save(str(self.vector_store_dir))

        return {
            "num_docs": len(docs),
            "num_chunks": len(chunks),
            "embedding_dim": embeddings.shape[1],
        }

    def load_index(self) -> Dict:
        """
        Load existing vector store.
        """

        self.vector_store.load(str(self.vector_store_dir))

        sources = sorted(
            list(
                {
                    chunk["source"]
                    for chunk in self.vector_store.chunks
                }
            )
        )

        return {
            "num_docs": len(sources),
            "num_chunks": len(self.vector_store.chunks),
            "embedding_dim": self.vector_store.embeddings.shape[1],
            "sources": sources,
        }

    def build_or_load_index(self) -> Dict:
        """
        Load vector store if it exists, otherwise build it.
        """

        if self.vector_store_exists():
            return self.load_index()

        build_info = self.build_index()
        load_info = self.load_index()

        return {
            **build_info,
            **load_info,
            "built_new_index": True,
        }

    def answer_query(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict], str]:
        """
        Retrieve relevant chunks and generate answer.
        """

        retriever = Retriever(self.vector_store, self.embedder)

        retrieved_chunks = retriever.retrieve(query, top_k=top_k)

        prompt = build_prompt(query, retrieved_chunks)

        fallback_context = "\n\n".join(
            [
                f"[Source: {chunk['source']} | Score: {chunk['score']:.4f}]\n"
                f"{chunk['text']}"
                for chunk in retrieved_chunks
            ]
        )

        answer = self.generator.generate(
            prompt=prompt,
            fallback_context=fallback_context,
        )

        return answer, retrieved_chunks, prompt