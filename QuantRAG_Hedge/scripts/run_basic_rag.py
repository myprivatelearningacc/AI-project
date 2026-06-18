import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.document_loader import load_markdown_docs
from src.chunker import chunk_documents
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.prompt_builder import build_prompt
from src.generator import Generator


RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "processed" / "basic_vector_store"


def build_index():
    print("Loading documents...")
    docs = load_markdown_docs(str(RAW_DOCS_DIR))
    print(f"Loaded {len(docs)} documents.")

    print("Chunking documents...")
    chunks = chunk_documents(docs, chunk_size=700, overlap=120)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks...")
    embedder = Embedder()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_texts(texts)
    print(f"Embedding shape: {embeddings.shape}")

    print("Saving vector store...")
    vector_store = VectorStore()
    vector_store.build(chunks, embeddings)
    vector_store.save(str(VECTOR_STORE_DIR))

    print(f"Vector store saved to: {VECTOR_STORE_DIR}")


def ask(query: str, top_k: int = 3):
    print("Loading embedder and vector store...")

    embedder = Embedder()

    vector_store = VectorStore()
    vector_store.load(str(VECTOR_STORE_DIR))

    retriever = Retriever(vector_store, embedder)

    print(f"\nQuery: {query}")
    print("\nRetrieving relevant chunks...")
    retrieved_chunks = retriever.retrieve(query, top_k=top_k)

    print("\nTop retrieved chunks:")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        print("-" * 80)
        print(f"Rank {i}")
        print(f"Source: {chunk['source']}")
        print(f"Score: {chunk['score']:.4f}")
        print(chunk["text"][:500])

    prompt = build_prompt(query, retrieved_chunks)

    fallback_context = "\n\n".join(
        [
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    generator = Generator()
    answer = generator.generate(prompt, fallback_context=fallback_context)

    print("\n" + "=" * 80)
    print("Generated Answer")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    build_index()

    sample_query = "What is delta hedging and why is it used in options trading?"
    ask(sample_query, top_k=3)