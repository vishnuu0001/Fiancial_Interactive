import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages downloading and storing models locally"""

    def __init__(self, model_path: str = "./models"):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)

    def download_model(self, model_name: str) -> bool:
        """Download and save model locally"""
        local_path = os.path.join(self.model_path, model_name.replace('/', '_'))

        if os.path.exists(local_path):
            logger.info(f"Model {model_name} already exists at {local_path}")
            return True

        try:
            logger.info(f"Downloading {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True
            )

            # Add padding token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)

            logger.info(f"Model saved to {local_path}")
            return True

        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False

    def list_available_models(self) -> List[str]:
        """List all locally available models"""
        if not os.path.exists(self.model_path):
            return []

        models = []
        for item in os.listdir(self.model_path):
            model_dir = os.path.join(self.model_path, item)
            if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
                models.append(item.replace('_', '/'))

        return models

    def download_recommended_models(self):
        """Download recommended models for financial document analysis"""
        models = [
            "microsoft/DialoGPT-small",  # Fast, good for testing
            "facebook/blenderbot-400M-distill",  # Good conversational model
            "microsoft/DialoGPT-medium",  # Better quality responses
        ]

        for model in models:
            self.download_model(model)


if __name__ == "__main__":
    manager = ModelManager()
    manager.download_recommended_models()