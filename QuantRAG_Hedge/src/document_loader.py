from pathlib import Path
from typing import List, Dict


def load_markdown_docs(raw_docs_dir: str) -> List[Dict]:
    """
    Load all markdown documents from a folder.

    Returns a list of documents, where each document contains:
    - doc_id
    - source
    - text
    """

    raw_path = Path(raw_docs_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Directory not found: {raw_docs_dir}")

    docs = []

    for file_path in sorted(raw_path.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")

        docs.append(
            {
                "doc_id": file_path.stem,
                "source": file_path.name,
                "text": text,
            }
        )

    if len(docs) == 0:
        raise ValueError(f"No markdown files found in {raw_docs_dir}")

    return docs