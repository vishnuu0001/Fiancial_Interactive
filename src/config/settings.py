import logging

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# RAG System configuration
TENK_FOLDER = "tenk"
MODEL_NAME = "llama2:latest"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5

# Vector Store configuration
COLLECTION_NAME = "pdf_chunks"
PERSIST_DIRECTORY = "./data/chromadb"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_TOKENS = 500
TEMPERATURE = 0.3