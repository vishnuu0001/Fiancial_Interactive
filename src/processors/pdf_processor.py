import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction and preprocessing"""

    def __init__(self):
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))

    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', ' ', text)
        return text.strip()

    def extract_text(self, pdf_path: str) -> str:
        """Extracts text from a PDF file."""
        text = ""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return text

    def split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
        """Splits text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
        return chunks

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if overlap > 0 and chunks:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + sentence + " "
                else:
                    current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks