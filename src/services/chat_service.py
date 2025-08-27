# Update src/services/chat_service.py
import logging
from typing import Dict, List, Any
from src.models.offline_llm import OfflineLLM
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, vector_store: VectorStore, llm_model_name: str = "microsoft/DialoGPT-medium"):
        self.vector_store = vector_store
        self.llm = OfflineLLM(model_name=llm_model_name)

    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process user query and return response with sources"""
        try:
            # Search for relevant documents
            search_results = self.vector_store.search(query, top_k=top_k)

            if not search_results:
                return {
                    'response': "I couldn't find relevant information in the documents. Please try rephrasing your question.",
                    'sources': []
                }

            # Filter results by relevance score
            filtered_results = [r for r in search_results if r.get('score', 0) > 0.1]

            if not filtered_results:
                return {
                    'response': "I found some documents but they don't seem directly relevant to your question. Please try a more specific query.",
                    'sources': []
                }

            # Generate response based on query type
            if self._is_financial_query(query):
                response = self._generate_financial_response(filtered_results, query)
            else:
                response = self._generate_general_response(filtered_results, query)

            # Extract sources
            sources = self._extract_sources(filtered_results)

            return {
                'response': response,
                'sources': sources
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'response': f"An error occurred while processing your query: {str(e)}",
                'sources': []
            }

    def _is_financial_query(self, query: str) -> bool:
        """Check if query is financial-related"""
        financial_keywords = [
            'revenue', 'profit', 'income', 'earnings', 'sales', 'cash',
            'assets', 'liabilities', 'equity', 'debt', 'financial',
            'balance sheet', 'income statement', 'cash flow', 'margin',
            'growth', 'performance', 'apple', '$', 'million', 'billion'
        ]
        return any(keyword in query.lower() for keyword in financial_keywords)

    def _generate_financial_response(self, results: List[Dict], query: str) -> str:
        """Generate financial-specific response"""
        # Prepare context from search results
        context_parts = []
        for result in results[:3]:  # Use top 3 most relevant
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            file_name = metadata.get('file_name', 'Unknown')
            page = metadata.get('page', 'Unknown')

            context_parts.append(f"From {file_name} (Page {page}):\n{text}\n")

        context = "\n".join(context_parts)

        # Create a focused financial prompt
        prompt = f"""Based on the following financial document excerpts, provide a direct and accurate answer to the question. Be specific with numbers and cite page references when possible.

Context from Apple's 10-K Filing:
{context}

Question: {query}

Please provide a clear, factual answer based only on the information provided above. If specific financial figures are mentioned, include them with proper formatting (e.g., $394.3 billion). Do not add speculative information."""

        try:
            response = self.llm.generate_response(prompt)
            return self._clean_response(response)
        except Exception as e:
            logger.error(f"Error generating financial response: {e}")
            return self._create_fallback_response(results, query)

    def _generate_general_response(self, results: List[Dict], query: str) -> str:
        """Generate general response for non-financial queries"""
        # Prepare context
        context_parts = []
        for result in results[:3]:
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            file_name = metadata.get('file_name', 'Unknown')
            page = metadata.get('page', 'Unknown')

            context_parts.append(f"From {file_name} (Page {page}):\n{text}\n")

        context = "\n".join(context_parts)

        prompt = f"""Based on the following document excerpts, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Provide a clear answer based on the information above."""

        try:
            response = self.llm.generate_response(prompt)
            return self._clean_response(response)
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            return self._create_fallback_response(results, query)

    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        # Remove common LLM artifacts
        response = response.strip()

        # Remove repetitive headers or generic phrases
        lines_to_remove = [
            "## Executive Summary",
            "Based on Apple's 10-K filings, here is a comprehensive financial analysis:",
            "# Apple Inc. Financial Analysis"
        ]

        for line in lines_to_remove:
            if response.startswith(line):
                response = response[len(line):].strip()

        # Ensure response doesn't start with generic templates
        if response.startswith("Based on") and "comprehensive" in response[:100]:
            # Find the first substantive sentence
            sentences = response.split('.')
            for i, sentence in enumerate(sentences):
                if any(keyword in sentence.lower() for keyword in ['revenue', 'profit', 'sales', 'cash', '$']):
                    response = '.'.join(sentences[i:]).strip()
                    break

        return response

    def _create_fallback_response(self, results: List[Dict], query: str) -> str:
        """Create a fallback response when LLM fails"""
        if not results:
            return "No relevant information found in the documents."

        # Extract key information directly from the most relevant result
        top_result = results[0]
        text = top_result.get('text', '')
        metadata = top_result.get('metadata', {})
        file_name = metadata.get('file_name', 'Unknown')
        page = metadata.get('page', 'Unknown')

        # Look for financial figures in the text
        import re
        financial_patterns = [
            r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?',
            r'revenue.*?\$[\d,]+',
            r'income.*?\$[\d,]+',
            r'sales.*?\$[\d,]+'
        ]

        found_figures = []
        for pattern in financial_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_figures.extend(matches)

        response = f"Based on {file_name} (Page {page}):\n\n"

        if found_figures:
            response += f"Key financial information found: {', '.join(found_figures[:3])}\n\n"

        # Include relevant excerpt
        response += f"Relevant excerpt: {text[:300]}..."

        return response

    def _extract_sources(self, results: List[Dict]) -> List[Dict]:
        """Extract source information from search results"""
        sources = []
        for result in results:
            metadata = result.get('metadata', {})
            sources.append({
                'file_name': metadata.get('file_name', 'Unknown'),
                'page': metadata.get('page', 'Unknown'),
                'score': result.get('score', 0.0),
                'chunk_index': metadata.get('chunk_index', 0)
            })
        return sources