#!/usr/bin/env python3
"""Setup script to download and configure offline models"""

import os
import sys
import logging

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.models.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Download and setup offline models"""

    print("Setting up offline models for financial document chat...")

    # Create models directory in project root
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Initialize model manager
    manager = ModelManager(model_path=models_dir)

    # Download recommended models
    print("\nDownloading recommended models...")
    manager.download_recommended_models()

    # List available models
    available_models = manager.list_available_models()
    print(f"\nAvailable models: {available_models}")

    print("\nSetup complete! You can now run the application offline.")

if __name__ == "__main__":
    main()