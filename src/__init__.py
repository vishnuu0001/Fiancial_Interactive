"""
RAG System with Ollama Llama2
A comprehensive Retrieval-Augmented Generation system for PDF document processing
"""

__version__ = "1.0.0"
__author__ = "Vishnuu"
__email__ = "vishnuu_a@yahoo.com"

from .rag.rag_system import RAGSystem
from .models.ollama_llm import OllamaLLM
from .processors.pdf_processor import PDFProcessor
from .storage.vector_store import VectorStore
from .retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    'RAGSystem',
    'OllamaLLM',
    'PDFProcessor',
    'VectorStore',
    'HybridRetriever'
]