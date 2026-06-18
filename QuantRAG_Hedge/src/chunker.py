from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    """
    Split text into overlapping chunks.

    chunk_size: approximate number of characters per chunk.
    overlap: number of characters shared between adjacent chunks.
    """

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_documents(
    docs: List[Dict],
    chunk_size: int = 700,
    overlap: int = 120
) -> List[Dict]:
    """
    Convert full documents into smaller chunks with metadata.
    """

    all_chunks = []

    for doc in docs:
        chunks = chunk_text(
            text=doc["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "text": chunk,
                }
            )

    return all_chunks