#!/usr/bin/env python3
"""Setup script to download and configure offline models"""

import os
import sys
import logging

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from src.models.model_manager import ModelManager
except ImportError:
    # Fallback: create ModelManager inline
    from transformers import AutoTokenizer, AutoModelForCausalLM


    class ModelManager:
        def __init__(self, model_path: str = "./models"):
            self.model_path = model_path
            os.makedirs(model_path, exist_ok=True)

        def download_model(self, model_name: str) -> bool:
            local_path = os.path.join(self.model_path, model_name.replace('/', '_'))

            if os.path.exists(local_path):
                print(f"Model {model_name} already exists")
                return True

            try:
                print(f"Downloading {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                tokenizer.save_pretrained(local_path)
                model.save_pretrained(local_path)
                print(f"Model saved to {local_path}")
                return True
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
                return False

        def download_recommended_models(self):
            models = ["microsoft/DialoGPT-small", "facebook/blenderbot-400M-distill", "microsoft/DialoGPT-medium"]
            for model in models:
                self.download_model(model)

logging.basicConfig(level=logging.INFO)


def main():
    print("Setting up offline models for financial document chat...")

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    manager = ModelManager(model_path=models_dir)
    manager.download_recommended_models()

    print("\nSetup complete!")


if __name__ == "__main__":
    main()