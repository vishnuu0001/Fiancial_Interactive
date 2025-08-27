from .services.chat_service import ChatService
from .storage.vector_store import VectorStore
from .models.model_manager import ModelManager

# First time setup - download models
manager = ModelManager()
manager.download_recommended_models()

# Initialize services
vector_store = VectorStore()
chat_service = ChatService(
    vector_store=vector_store,
    llm_model_name="microsoft/DialoGPT-medium"
)

# Ask questions
response = chat_service.ask_question("What was the revenue growth?")
print(response["answer"])