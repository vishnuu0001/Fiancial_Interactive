# Update src/models/offline_llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

logger = logging.getLogger(__name__)

class OfflineLLM:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing OfflineLLM with {model_name} on {self.device}")

        try:
            # Use a better model for financial analysis
            if "DialoGPT" in model_name:
                # Switch to a more suitable model for Q&A
                self.model_name = "distilbert-base-cased-distilled-squad"
                self.pipeline = pipeline(
                    "question-answering",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=0 if self.device == 'cuda' else -1
                )
                self.mode = "qa"
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(self.device)
                self.mode = "generative"

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to simple text processing
            self.mode = "fallback"

    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response based on the prompt"""
        try:
            if self.mode == "qa":
                return self._qa_response(prompt)
            elif self.mode == "generative":
                return self._generative_response(prompt, max_length)
            else:
                return self._fallback_response(prompt)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._fallback_response(prompt)

    def _qa_response(self, prompt: str) -> str:
        """Use Q&A pipeline for better accuracy"""
        try:
            # Extract question and context from prompt
            parts = prompt.split("Question:")
            if len(parts) < 2:
                return "Please provide a clear question."

            question = parts[1].strip()
            context_part = parts[0].replace("Context:", "").replace("Context from Apple's 10-K Filing:", "").strip()

            if len(context_part) > 3000:  # Truncate if too long
                context_part = context_part[:3000]

            result = self.pipeline(question=question, context=context_part)

            confidence = result.get('score', 0)
            answer = result.get('answer', '')

            if confidence > 0.1 and answer:
                return answer
            else:
                return "I couldn't find a confident answer in the provided context."

        except Exception as e:
            logger.error(f"Error in QA response: {e}")
            return self._fallback_response(prompt)

    def _generative_response(self, prompt: str, max_length: int) -> str:
        """Use generative model"""
        try:
            # Tokenize and generate
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the original prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()

            return response

        except Exception as e:
            logger.error(f"Error in generative response: {e}")
            return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when models fail"""
        # Extract key information from prompt using simple text processing
        import re

        # Look for financial figures
        financial_figures = re.findall(r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?', prompt)

        # Look for key terms
        key_terms = []
        financial_keywords = ['revenue', 'profit', 'income', 'sales', 'cash', 'assets', 'growth']
        for keyword in financial_keywords:
            if keyword in prompt.lower():
                # Extract sentence containing the keyword
                sentences = prompt.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        key_terms.append(sentence.strip())
                        break

        response = ""
        if financial_figures:
            response += f"Financial figures mentioned: {', '.join(financial_figures[:3])}\n\n"

        if key_terms:
            response += f"Key information: {'. '.join(key_terms[:2])}"

        if not response:
            response = "I found relevant information in the documents, but need a more specific question to provide accurate details."

        return response