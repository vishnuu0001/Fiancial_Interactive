import os
import sys
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.storage.document_processor import DocumentProcessor
from src.storage.vector_store import VectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reprocess_documents():
    """Reprocess all PDF documents with proper page numbers"""

    # Initialize components
    doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    vector_store = VectorStore(collection_name="pdf_chunks", persist_directory="./data/chromadb")

    # Path to your PDF documents - UPDATED TO TENK FOLDER
    pdf_directory = "./data/tenk"  # Updated path

    if not os.path.exists(pdf_directory):
        logger.error(f"PDF directory not found: {pdf_directory}")
        return

    # Clear existing collection to avoid duplicates
    try:
        vector_store.client.delete_collection("pdf_chunks")
        logger.info("Deleted existing collection")
    except:
        logger.info("No existing collection to delete")

    # Recreate collection
    vector_store.collection = vector_store.client.create_collection("pdf_chunks")
    logger.info("Created new collection")

    # Process all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_directory}")
        return

    total_chunks = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        logger.info(f"Processing: {pdf_file}")

        try:
            # Process PDF and get chunks with metadata
            chunks_data = doc_processor.process_pdf(pdf_path)

            if chunks_data:
                # Separate chunks and metadata
                chunks = [chunk_data['text'] for chunk_data in chunks_data]
                metadata = [chunk_data['metadata'] for chunk_data in chunks_data]

                # Add to vector store
                vector_store.add_chunks(chunks, metadata)

                total_chunks += len(chunks)
                logger.info(f"Added {len(chunks)} chunks from {pdf_file}")
            else:
                logger.warning(f"No chunks extracted from {pdf_file}")

        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")

    logger.info(f"Reprocessing complete! Total chunks: {total_chunks}")

    # Verify the reprocessing
    verify_reprocessing(vector_store)

def verify_reprocessing(vector_store):
    """Verify that documents have proper page numbers"""
    logger.info("Verifying reprocessed documents...")

    # Test search to check if page numbers are present
    test_results = vector_store.search("Apple financial", top_k=3)

    for i, result in enumerate(test_results, 1):
        metadata = result.get('metadata', {})
        page = metadata.get('page', 'Unknown')
        file_name = metadata.get('file_name', 'Unknown')
        score = result.get('score', 0)

        logger.info(f"Sample {i}: {file_name}, Page: {page}, Score: {score:.3f}")

        if page == 'Unknown':
            logger.warning("Page numbers still showing as 'Unknown' - check document processor")
        else:
            logger.info("✓ Page numbers are working correctly")

if __name__ == "__main__":
    reprocess_documents()