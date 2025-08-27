import sys
import os

from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


import numpy as np
import logging
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

# Use absolute imports
from src.models.ollama_llm import OllamaLLM
from src.processors.pdf_processor import PDFProcessor
from src.storage.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system that orchestrates the entire pipeline"""

    def __init__(self, tenk_folder: str, model_name: str = "llama2"):
        # Force CUDA usage
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        self.tenk_folder = Path(tenk_folder)
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.llm = OllamaLLM(model_name=model_name)
        self.documents_processed = False

    # ... rest of your existing code
    def process_pdfs(self):
        """Process all PDF files in the tenk folder"""
        if not self.tenk_folder.exists():
            logger.error(f"Folder {self.tenk_folder} does not exist")
            return

        pdf_files = list(self.tenk_folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.tenk_folder}")
            return

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        all_chunks = []
        all_metadata = []

        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}")

            # Extract text from PDF
            text = self.pdf_processor.extract_text(str(pdf_file))

            if not text.strip():
                logger.warning(f"No text extracted from {pdf_file.name}")
                continue

            # Split text into chunks
            chunks = self.pdf_processor.split_text(text)

            # Create metadata for each chunk
            metadata = [
                {
                    'file_name': pdf_file.name,
                    'chunk_index': i,
                    'file_path': str(pdf_file)
                }
                for i in range(len(chunks))
            ]

            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

        if all_chunks:
            # Add chunks to vector store
            self.vector_store.add_chunks(all_chunks, all_metadata)

            # Initialize BM25 for hybrid retrieval
            self.retriever.initialize_bm25(all_chunks, all_metadata)

            self.documents_processed = True
            logger.info(f"Successfully processed {len(all_chunks)} chunks from {len(pdf_files)} files")
        else:
            logger.error("No chunks were extracted from any PDF files")

    # In src/rag/rag_system.py - update the query method
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.documents_processed:
            logger.error("No documents have been processed yet. Call process_pdfs() first.")
            return {
                'question': question,
                'answer': "No documents have been processed yet.",
                'confidence': 0.0,
                'sources': []
            }

        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, k=5)

        if not retrieved_docs:
            return {
                'question': question,
                'answer': "I couldn't find relevant information to answer your question.",
                'confidence': 0.0,
                'sources': []
            }

        # Filter out low-scoring results
        filtered_docs = [doc for doc in retrieved_docs if doc['final_score'] > 0.1]

        if not filtered_docs:
            filtered_docs = retrieved_docs[:3]  # Use top 3 if all scores are low

        # Create context from retrieved documents
        context = "\n\n".join([doc['document'] for doc in filtered_docs])

        # Generate answer using LLM
        prompt = f"""Based on the following context, please answer the question accurately and concisely.

    Context:
    {context}

    Question: {question}

    Answer:"""

        answer = self.llm.generate(prompt)

        # Calculate confidence based on retrieval scores and answer quality
        avg_score = sum(doc['final_score'] for doc in filtered_docs) / len(filtered_docs)

        # Additional confidence factors
        answer_length_factor = min(len(answer.split()) / 20, 1.0)  # Longer answers get slight boost
        context_relevance = min(len(filtered_docs) / 5, 1.0)  # More relevant docs = higher confidence

        final_confidence = avg_score * 0.7 + answer_length_factor * 0.15 + context_relevance * 0.15

        return {
            'question': question,
            'answer': answer,
            'confidence': min(final_confidence, 1.0),
            'sources': [doc['metadata'] for doc in filtered_docs],
            'retrieval_scores': [
                {
                    'file': doc['metadata']['file_name'],
                    'vector_score': doc['vector_score'],
                    'bm25_score': doc['bm25_score'],
                    'final_score': doc['final_score']
                }
                for doc in filtered_docs
            ]
        }

    def chat(self):
        """Interactive chat interface"""
        if not self.documents_processed:
            logger.error("No documents have been processed yet. Call process_pdfs() first.")
            return

        print("RAG System Chat Interface")
        print("Type 'quit' or 'exit' to end the chat")
        print("-" * 50)

        while True:
            question = input("\nYour question: ").strip()

            if question.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            if not question:
                continue

            result = self.query(question)

            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Sources: {len(result['sources'])} documents")