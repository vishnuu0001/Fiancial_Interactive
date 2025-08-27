import requests
import logging

logger = logging.getLogger(__name__)

class OllamaLLM:
    """Interface for Ollama Llama2 model"""

    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self._test_connection()

    def _test_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                    logger.info(f"Run 'ollama pull {self.model_name}' to download the model")
                else:
                    logger.info(f"Connected to Ollama. Model {self.model_name} is available.")
            else:
                logger.error("Failed to connect to Ollama API")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection failed: {e}")
            logger.info("Make sure Ollama is running with 'ollama serve'")

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        """Generate response using Ollama"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
            }
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "Error: Unable to generate response from LLM"

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return "Error: Unable to connect to LLM service"