import PyPDF2
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Process PDF and extract chunks with proper page numbers"""
        chunks = []

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()

                    if page_text.strip():
                        # Split page text into chunks
                        page_chunks = self._split_text_into_chunks(page_text)

                        for chunk_idx, chunk in enumerate(page_chunks):
                            chunks.append({
                                'text': chunk,
                                'metadata': {
                                    'file_name': file_path.split('/')[-1],
                                    'page': page_num,  # Actual page number
                                    'chunk_index': chunk_idx,
                                    'total_pages': len(pdf_reader.pages)
                                }
                            })

            logger.info(f"Processed {len(chunks)} chunks from {len(pdf_reader.pages)} pages")
            return chunks

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks