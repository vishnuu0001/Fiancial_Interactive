from pathlib import Path
import numpy as np
import logging
from typing import Dict, Any

from ..models.ollama_llm import OllamaLLM
from ..processors.pdf_processor import PDFProcessor
from ..storage.vector_store import VectorStore
from ..retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RAGSystem:
    """Main RAG system that orchestrates the entire pipeline"""

    def __init__(self, tenk_folder: str, model_name: str = "llama2"):
        self.tenk_folder = Path(tenk_folder)
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.llm = OllamaLLM(model_name=model_name)
        self.documents_processed = False

    def process_pdfs(self):
        """Process all PDFs in the tenk folder"""
        if not self.tenk_folder.exists():
            raise FileNotFoundError(f"Folder {self.tenk_folder} does not exist")

        pdf_files = list(self.tenk_folder.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.tenk_folder}")

        all_chunks = []
        all_metadata = []
        all_documents = []

        logger.info(f"Processing {len(pdf_files)} PDF files...")

        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}...")
            text = self.pdf_processor.extract_text_from_pdf(str(pdf_file))

            if not text:
                logger.warning(f"No text extracted from {pdf_file.name}")
                continue

            cleaned_text = self.pdf_processor.clean_text(text)
            chunks = self.pdf_processor.chunk_text(cleaned_text)

            for i, chunk in enumerate(chunks):
                metadata = {
                    'file_name': pdf_file.name,
                    'file_path': str(pdf_file),
                    'chunk_index': i,
                    'chunk_id': f"{pdf_file.stem}_{i}",
                    'total_chunks': len(chunks)
                }
                all_chunks.append(chunk)
                all_metadata.append(metadata)
                all_documents.append(chunk)

        if all_chunks:
            self.vector_store.add_chunks(all_chunks, all_metadata)
            self.retriever.build_bm25_index(all_documents, all_metadata)
            self.documents_processed = True
            logger.info(f"Successfully processed {len(all_chunks)} chunks from {len(pdf_files)} PDF files")
        else:
            raise ValueError("No chunks were created from the PDF files")

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.documents_processed:
            raise ValueError("No documents have been processed. Call process_pdfs() first.")

        retrieved_chunks = self.retriever.retrieve(question, k=top_k)

        if not retrieved_chunks:
            return {
                'question': question,
                'answer': "No relevant information found in the documents.",
                'sources': [],
                'confidence': 0.0
            }

        context = "\n\n".join([chunk['document'] for chunk in retrieved_chunks])
        top_scores = [chunk['combined_score'] for chunk in retrieved_chunks[:3]]
        confidence = np.mean(top_scores) if top_scores else 0.0

        sources = []
        for chunk in retrieved_chunks:
            source_info = {
                'file_name': chunk['metadata']['file_name'],
                'chunk_index': chunk['metadata']['chunk_index'],
                'relevance_score': round(chunk['combined_score'], 3)
            }
            if source_info not in sources:
                sources.append(source_info)

        answer = self._generate_answer_with_llm(question, context, confidence)

        return {
            'question': question,
            'answer': answer,
            'context': context,
            'sources': sources,
            'confidence': round(confidence, 3),
            'num_chunks_used': len(retrieved_chunks)
        }

    def _generate_answer_with_llm(self, question: str, context: str, confidence: float) -> str:
        """Generate answer using Ollama Llama2 model"""
        if confidence < 0.3:
            return f"Based on the available documents, I found limited relevant information for your question: '{question}'. The confidence in this response is low ({confidence:.2f})."

        prompt = f"""You are a helpful AI assistant that answers questions based on provided context from PDF documents.

Context from PDF documents:
{context}

Question: {question}

Instructions:
- Answer based solely on the provided context
- If the context doesn't contain enough information, say so clearly
- Be specific and cite relevant details from the context
- Keep your answer concise but complete

Answer:"""

        try:
            response = self.llm.generate(prompt=prompt, max_tokens=500, temperature=0.3)

            if response and not response.startswith("Error:"):
                confidence_text = ""
                if confidence < 0.5:
                    confidence_text = " (Note: Moderate confidence)"
                elif confidence < 0.7:
                    confidence_text = " (Note: Good confidence)"
                else:
                    confidence_text = " (Note: High confidence)"

                return response + confidence_text
            else:
                return response or "Unable to generate a response at this time."

        except Exception as e:
            logger.error(f"Error generating answer with LLM: {str(e)}")
            return f"An error occurred while generating the answer: {str(e)}"

    def chat(self):
        """Interactive chat interface"""
        print("RAG System with Llama2 is ready! Type 'quit' to exit.")
        print("-" * 50)

        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                print("\nProcessing your question...")
                result = self.query(question)

                print(f"\nAnswer: {result['answer']}")
                print(f"\nConfidence: {result['confidence']}")
                print(f"Sources: {len(result['sources'])} files")

                show_sources = input("\nShow sources? (y/n): ").strip().lower()
                if show_sources == 'y':
                    print("\nSources:")
                    for source in result['sources']:
                        print(
                            f"- {source['file_name']} (chunk {source['chunk_index']}, relevance: {source['relevance_score']})")

                print("-" * 50)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")