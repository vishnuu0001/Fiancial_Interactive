# src/retrieval/hybrid_retriever.py
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, vector_store, bm25_weight=0.5, vector_weight=0.5):
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25 = None
        self.documents = None
        self.metadata = None

    def initialize_bm25(self, chunks, metadata):
        """Initialize BM25 with document chunks and metadata"""
        self.documents = chunks
        self.metadata = metadata
        tokenized_docs = [word_tokenize(doc.lower()) for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"BM25 initialized with {len(chunks)} documents")

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval combining vector similarity and BM25"""
        # Get vector search results
        vector_results = self.vector_store.similarity_search(query, k=k * 2)

        # Get BM25 results if initialized
        bm25_results = []
        if self.bm25 and self.documents:
            tokenized_query = word_tokenize(query.lower())
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Get top k BM25 results
            top_indices = np.argsort(bm25_scores)[::-1][:k * 2]
            bm25_results = [
                {
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'bm25_score': float(bm25_scores[idx])
                }
                for idx in top_indices if bm25_scores[idx] > 0
            ]

        # Combine and score results
        combined_results = self._combine_results(vector_results, bm25_results, query)

        # Return top k results
        return sorted(combined_results, key=lambda x: x['final_score'], reverse=True)[:k]

    def _combine_results(self, vector_results, bm25_results, query):
        """Combine vector and BM25 results with hybrid scoring"""
        doc_scores = {}

        # Process vector results
        for result in vector_results:
            doc_text = result['document']
            vector_score = result.get('similarity_score', 0)

            doc_scores[doc_text] = {
                'document': doc_text,
                'metadata': result['metadata'],
                'vector_score': vector_score,
                'bm25_score': 0.0
            }

        # Process BM25 results
        max_bm25_score = max([r['bm25_score'] for r in bm25_results]) if bm25_results else 1.0

        for result in bm25_results:
            doc_text = result['document']
            normalized_bm25 = result['bm25_score'] / max_bm25_score if max_bm25_score > 0 else 0

            if doc_text in doc_scores:
                doc_scores[doc_text]['bm25_score'] = normalized_bm25
            else:
                doc_scores[doc_text] = {
                    'document': doc_text,
                    'metadata': result['metadata'],
                    'vector_score': 0.0,
                    'bm25_score': normalized_bm25
                }

        # Calculate final hybrid scores
        final_results = []
        for doc_info in doc_scores.values():
            # Apply keyword relevance boost
            keyword_boost = self._calculate_keyword_boost(doc_info['document'], query)

            final_score = (
                    self.vector_weight * doc_info['vector_score'] +
                    self.bm25_weight * doc_info['bm25_score'] +
                    keyword_boost
            )

            final_results.append({
                'document': doc_info['document'],
                'metadata': doc_info['metadata'],
                'vector_score': doc_info['vector_score'],
                'bm25_score': doc_info['bm25_score'],
                'keyword_boost': keyword_boost,
                'final_score': min(final_score, 1.0)  # Cap at 1.0
            })

        return final_results

    def _calculate_keyword_boost(self, document: str, query: str) -> float:
        """Calculate keyword matching boost"""
        doc_lower = document.lower()
        query_words = query.lower().split()

        matches = sum(1 for word in query_words if word in doc_lower)
        keyword_boost = matches / len(query_words) * 0.2  # Max 0.2 boost

        return keyword_boost