from typing import List, Dict


def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build a RAG prompt using retrieved context.
    """

    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[Context {i} | Source: {chunk['source']} | Score: {chunk['score']:.4f}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful quantitative finance tutor.

Use ONLY the provided context to answer the question.
If the context is insufficient, say that the provided documents do not contain enough information.

Question:
{query}

Context:
{context}

Answer:
""".strip()

    return prompt