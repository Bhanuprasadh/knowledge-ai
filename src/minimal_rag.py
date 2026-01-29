import os
from pathlib import Path
from typing import List, Tuple
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer

import faiss
from groq import Groq

# Load Documents

def load_documents(data_dir: str = "data/raw") -> List[str]:
    documents = []
    data_path = Path(data_dir)

    for file_path in data_path.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append(content)
            print(f" Loaded: {file_path.name}")

    print(f"Total documents loaded: {len(documents)}")
    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())

        start += (chunk_size - overlap)
        
    return chunks