#!/usr/bin/env python3
"""Main entry point for the financial document chat application"""

import os
import sys
import logging

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.services.chat_service import ChatService
from src.storage.vector_store import VectorStore
from src.models.model_manager import ModelManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Update your main.py to test the improvements
def main():
    logger.info("Starting Financial Chat Bot")

    try:
        # Initialize vector store
        vector_store = VectorStore()

        # Check if we have documents
        if not vector_store.collection.count():
            print("No documents found in vector store. Please run document processing first.")
            return

        # Initialize chat service with Q&A model
        chat_service = ChatService(
            vector_store=vector_store,
            llm_model_name="distilbert-base-cased-distilled-squad"
        )

        # Interactive chat loop
        print("Financial Chat Bot is ready! Type 'quit' to exit.")
        print("Try asking specific questions like:")
        print("- What was Apple's total revenue in 2022?")
        print("- What are Apple's main sources of income?")
        print("- How much cash does Apple have?")

        while True:
            user_query = input("\nYour question: ")

            if user_query.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break

            if not user_query.strip():
                continue

            print("Searching documents...")

            # Process the query using ChatService
            result = chat_service.process_query(user_query, top_k=3)

            print(f"\nAnswer: {result['response']}")

            if result['sources']:
                print(f"\nSources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['file_name']} (Page: {source['page']}, Score: {source['score']:.3f})")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()