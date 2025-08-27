import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import logging
import torch
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector database operations using ChromaDB"""

    def __init__(self, collection_name: str = "pdf_chunks", persist_directory: str = "./data/chromadb"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name

        # Force CUDA usage - BEFORE model initialization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        # Initialize SentenceTransformer with GPU support
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            logger.info(f"Created new collection: {collection_name}")

    def add_chunks(self, chunks: List[str], metadata: List[Dict[str, Any]]):
        """Add text chunks to vector database"""
        embeddings = self.model.encode(chunks).tolist()
        ids = [hashlib.md5(f"{meta['file_name']}_{i}".encode()).hexdigest()
               for i, meta in enumerate(metadata)]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )
        logger.info(f"Added {len(chunks)} chunks to vector database")

    def similarity_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )

        return [
            {
                'document': doc,
                'metadata': meta,
                'similarity_score': 1 - dist
            }
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]

    def get_all_documents(self):
        """Get all documents from the collection"""
        results = self.collection.get()
        return [
            {
                'document': doc,
                'metadata': meta
            }
            for doc, meta in zip(results['documents'], results['metadatas'])
        ]

    def count(self) -> int:
        """Get total number of documents"""
        return self.collection.count()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using the query"""
        try:
            # Get query embedding
            query_embedding = self.model.encode([query])

            # Search in ChromaDB collection
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'score': 1 - results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0,
                        'id': results['ids'][0][i] if results['ids'] and results['ids'][0] else None
                    })

            logger.info(f"Found {len(formatted_results)} relevant documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []