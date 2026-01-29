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


# Create Embeddings

class EmbeddingModel:
    def __init__(self, model_name:str = "all-MiniLM-L6-v2"):
        print(f"Loading Embedding Model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded: Embedding Dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings


# Build Vector Store (FAISS)

class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        print(f" Created FAISS index with dimension: {dimension}")

    def add(self, embeddings: np.ndarray, chunks: List[str]):
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
        print(f" Added {len(chunks)} chunks to vector store")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append((self.chunks[idx], float(distance)))
        
        return results



# RAG Pipeline

class SimpleRAG:
    def __init__(self):
        print("Initializing RAG System...\n")
        self.embedding_model = EmbeddingModel()
        self.dimension = self.embedding_model.model.get_sentence_embedding_dimension()
        self.vector_store = VectorStore(self.dimension)

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")
        self.groq_client = Groq(api_key=api_key)
        print(" Groq client initialized\n")
    
    def ingest_documents(self, documents: List[str]):
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = chunk_text(doc)
            all_chunks.extend(chunks)
            print(f"Document {i+1}: Created {len(chunks)} chunks")
        
        print(f"\n Total chunks: {len(all_chunks)}")
        print(" Generating embeddings...")
        embeddings = self.embedding_model.encode(all_chunks)
        self.vector_store.add(embeddings, all_chunks)
        print(" Ingestion complete!\n")
    
    def ask(self, question: str, top_k: int = 3) -> str:
        print(f" Question: {question}\n")
        question_embedding = self.embedding_model.encode([question])
        results = self.vector_store.search(question_embedding, top_k)
        
        context = "\n\n".join([chunk for chunk, distance in results])
        prompt = f"""Based on the context, answer the question.
Context:
{context}
Question: {question}
Answer:"""
        
        print("🤖 Generating answer...\n")
        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        print("="*60)
        print("💡 ANSWER:")
        print("="*60)
        print(answer)
        print("="*60 + "\n")
        return answer

# MAIN


if __name__ == "__main__":
    print("\n MINIMAL RAG SYSTEM\n")
    
    documents = load_documents()
    if not documents:
        print("  No documents found!")
        exit(1)
    
    rag = SimpleRAG()
    rag.ingest_documents(documents)
    
    questions = [
        "What is Python?",
        "Explain machine learning",
    ]
    
    for question in questions:
        rag.ask(question)
        print("\n")