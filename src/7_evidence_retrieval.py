"""
Evidence Retrieval Module
Retrieves relevant evidence chunks for claims from paper content
"""

import json
import logging
import numpy as np
from typing import List, Dict, Any

# Try to import ML dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False
    logging.warning("TF-IDF dependencies not available.")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("BM25 dependencies not available.")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logging.warning("Sentence Transformers not available.")

class EvidenceRetriever:
    def __init__(self):
        """Initialize evidence retriever."""
        self.available_methods = []
        self.sbert_model = None  # Cache for SBERT model
        
        if TFIDF_AVAILABLE:
            self.available_methods.append("tfidf")
        
        if BM25_AVAILABLE:
            self.available_methods.append("bm25")
        
        if SBERT_AVAILABLE:
            self.available_methods.append("sbert")
        
        logging.info(f"Available evidence retrieval methods: {self.available_methods}")
    
    def load_chunks_from_jsonl(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """
        Load chunks from JSONL file.
        
        Args:
            jsonl_path (str): Path to JSONL file
        
        Returns:
            list: List of chunk dictionaries
        """
        chunks = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'chunks' in data:
                        chunks.extend(data['chunks'])
        
        return chunks
    
    def retrieve_evidence_tfidf(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve evidence using TF-IDF cosine similarity.
        
        Args:
            query (str): Claim query
            chunks (list): List of chunk dictionaries
            top_k (int): Number of top chunks to retrieve
        
        Returns:
            list: List of evidence dictionaries with chunk_id and text
        """
        if not TFIDF_AVAILABLE:
            raise ImportError("TF-IDF dependencies not available")
        
        if not chunks:
            return []
        
        # Extract text from chunks
        chunk_texts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
            else:
                text = str(chunk) if chunk else ""
            
            if text.strip():
                chunk_texts.append(text)
        
        if not chunk_texts:
            return []
        
        try:
            # Build corpus with query as first document
            corpus = [query] + chunk_texts
            
            # Configure TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                min_df=1,
                max_df=0.95,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode',
                analyzer='word'
            )
            
            # Fit and transform
            vectors = vectorizer.fit_transform(corpus)
            
            if vectors.shape[1] == 0:
                logging.warning(f"Empty vocabulary for query: {query[:50]}...")
                return []
            
            # Get query and chunk vectors
            query_vector = vectors[0]
            chunk_vectors = vectors[1:]
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, chunk_vectors)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return corresponding chunk texts
            return [chunk_texts[i] for i in top_indices]
            
        except Exception as e:
            logging.error(f"Error in TF-IDF retrieval: {e}")
            return []
    
    def retrieve_evidence_bm25(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
        """
        Retrieve evidence using BM25.
        
        Args:
            query (str): Claim query
            chunks (list): List of chunk dictionaries
            top_k (int): Number of top chunks to retrieve
        
        Returns:
            list: List of evidence texts
        """
        if not BM25_AVAILABLE:
            raise ImportError("BM25 dependencies not available")
        
        if not chunks:
            return []
        
        # Extract text from chunks
        chunk_texts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
            else:
                text = str(chunk) if chunk else ""
            
            if text.strip():
                chunk_texts.append(text)
        
        if not chunk_texts:
            return []
        
        try:
            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in chunk_texts]
            
            # Create BM25 object
            bm25 = BM25Okapi(tokenized_docs)
            
            # Search
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Return corresponding chunk texts
            return [chunk_texts[i] for i in top_indices]
            
        except Exception as e:
            logging.error(f"Error in BM25 retrieval: {e}")
            return []
    
    def retrieve_evidence_sbert(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
        """
        Retrieve evidence using Sentence Transformers.
        
        Args:
            query (str): Claim query
            chunks (list): List of chunk dictionaries
            top_k (int): Number of top chunks to retrieve
        
        Returns:
            list: List of evidence texts
        """
        if not SBERT_AVAILABLE:
            raise ImportError("Sentence Transformers not available")
        
        if not chunks:
            return []
        
        # Extract text from chunks
        chunk_texts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
            else:
                text = str(chunk) if chunk else ""
            
            if text.strip():
                chunk_texts.append(text)
        
        if not chunk_texts:
            return []
        
        try:
            # Load model (cache it to avoid re-downloading)
            if self.sbert_model is None:
                logging.info("Loading SentenceTransformer model (this will only happen once)...")
                self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Encode query and documents
            query_embedding = self.sbert_model.encode([query])
            doc_embeddings = self.sbert_model.encode(chunk_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return corresponding chunk texts
            return [chunk_texts[i] for i in top_indices]
            
        except Exception as e:
            logging.error(f"Error in SBERT retrieval: {e}")
            return []
    
    def retrieve_evidence(self, query: str, chunks: List[Dict[str, Any]], method: str = "auto", top_k: int = 3) -> dict:
        """
        Retrieve evidence for a claim using specified method.
        
        Args:
            query (str): Claim query
            chunks (list): List of chunk dictionaries
            method (str): Retrieval method ("auto", "tfidf", "bm25", "sbert")
            top_k (int): Number of top chunks to retrieve
        
        Returns:
            dict: Dictionary with evidence and actual method used
        """
        
        # Check if chunks are positional chunks (have 'id' field)
        is_positional = chunks and isinstance(chunks[0], dict) and 'id' in chunks[0]
        
        if is_positional:
            return self.retrieve_evidence_positional(query, chunks, method, top_k)
        else:
            return self.retrieve_evidence_legacy(query, chunks, method, top_k)
    
    def retrieve_evidence_positional(self, query: str, chunks: List[Dict[str, Any]], method: str = "auto", top_k: int = 3) -> dict:
        """
        Retrieve evidence from positional chunks, returning chunk IDs.
        
        Args:
            query (str): Claim query
            chunks (list): List of positional chunk dictionaries
            method (str): Retrieval method ("auto", "tfidf", "bm25", "sbert")
            top_k (int): Number of top chunks to retrieve
        
        Returns:
            dict: Dictionary with evidence (chunk IDs and text) and actual method used
        """
        original_method = method
        fallback_chain = []
        
        if method == "auto":
            # Try methods in order of preference
            for preferred_method in ["sbert", "bm25", "tfidf"]:
                if preferred_method in self.available_methods:
                    method = preferred_method
                    fallback_chain.append(preferred_method)
                    break
            else:
                raise ValueError("No evidence retrieval methods available")
        
        if method not in self.available_methods:
            raise ValueError(f"Method '{method}' not available. Available methods: {self.available_methods}")
        
        logging.info(f"Retrieving evidence using method: {method} (positional chunks)")
        
        # Extract text and IDs from positional chunks
        chunk_data = []
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("id", "")
            if text.strip() and chunk_id:
                chunk_data.append({
                    "id": chunk_id,
                    "text": text
                })
        
        if not chunk_data:
            return {
                'evidence': [],
                'actual_method': method,
                'configured_method': original_method,
                'fallback_chain': fallback_chain
            }
        
        # Use existing retrieval methods but return chunk IDs
        if method == "tfidf":
            evidence = self.retrieve_evidence_tfidf_positional(query, chunk_data, top_k)
        elif method == "bm25":
            evidence = self.retrieve_evidence_bm25_positional(query, chunk_data, top_k)
        elif method == "sbert":
            try:
                evidence = self.retrieve_evidence_sbert_positional(query, chunk_data, top_k)
            except Exception as e:
                logging.warning(f"SBERT failed, falling back to TF-IDF: {e}")
                evidence = self.retrieve_evidence_tfidf_positional(query, chunk_data, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'evidence': evidence,
            'actual_method': method,
            'configured_method': original_method,
            'fallback_chain': fallback_chain
        }
    
    def retrieve_evidence_legacy(self, query: str, chunks: List[Dict[str, Any]], method: str = "auto", top_k: int = 3) -> dict:
        """
        Retrieve evidence from legacy markdown chunks (original method).
        
        Args:
            query (str): Claim query
            chunks (list): List of chunk dictionaries
            method (str): Retrieval method ("auto", "tfidf", "bm25", "sbert")
            top_k (int): Number of top chunks to retrieve
        
        Returns:
            dict: Dictionary with evidence and actual method used
        """
        original_method = method
        fallback_chain = []
        
        if method == "auto":
            # Try methods in order of preference
            for preferred_method in ["sbert", "bm25", "tfidf"]:
                if preferred_method in self.available_methods:
                    method = preferred_method
                    fallback_chain.append(preferred_method)
                    break
            else:
                raise ValueError("No evidence retrieval methods available")
        
        if method not in self.available_methods:
            raise ValueError(f"Method '{method}' not available. Available methods: {self.available_methods}")
        
        logging.info(f"Retrieving evidence using method: {method}")
        
        if method == "tfidf":
            evidence = self.retrieve_evidence_tfidf(query, chunks, top_k)
        elif method == "bm25":
            evidence = self.retrieve_evidence_bm25(query, chunks, top_k)
        elif method == "sbert":
            evidence = self.retrieve_evidence_sbert(query, chunks, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'evidence': evidence,
            'actual_method': method,
            'configured_method': original_method,
            'fallback_chain': fallback_chain
        }
    
    def retrieve_evidence_for_claims(self, claims: List[str], chunks: List[Dict[str, Any]], 
                                   method: str = "auto", top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve evidence for multiple claims.
        
        Args:
            claims (list): List of claims
            chunks (list): List of chunk dictionaries
            method (str): Retrieval method
            top_k (int): Number of top chunks per claim
        
        Returns:
            list: List of claims with evidence
        """
        results = []
        
        for i, claim in enumerate(claims):
            evidence = self.retrieve_evidence(claim, chunks, method, top_k)
            
            result = {
                'claim_id': i + 1,
                'claim': claim,
                'evidence': evidence,
                'num_evidence': len(evidence)
            }
            results.append(result)
        
        return results
    
    def save_evidence_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save evidence retrieval results to JSON file.
        
        Args:
            results (list): List of claims with evidence
            output_path (str): Path to output JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        total_evidence = sum(len(result.get('evidence', [])) for result in results)
        logging.info(f"Saved {len(results)} claims with {total_evidence} evidence chunks to {output_path}")
    
    def retrieve_evidence_tfidf_positional(self, query: str, chunk_data: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve evidence using TF-IDF with positional chunks."""
        if not TFIDF_AVAILABLE:
            raise ImportError("TF-IDF dependencies not available")
        
        if not chunk_data:
            return []
        
        try:
            # Extract texts for TF-IDF
            chunk_texts = [item["text"] for item in chunk_data]
            corpus = [query] + chunk_texts
            
            # Configure TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                min_df=1,
                max_df=0.95,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode',
                analyzer='word'
            )
            
            # Fit and transform
            vectors = vectorizer.fit_transform(corpus)
            
            if vectors.shape[1] == 0:
                logging.warning(f"Empty vocabulary for query: {query[:50]}...")
                return []
            
            # Get query and chunk vectors
            query_vector = vectors[0]
            chunk_vectors = vectors[1:]
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, chunk_vectors)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return evidence with chunk IDs
            evidence = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    evidence.append({
                        "chunk_id": chunk_data[idx]["id"],
                        "text": chunk_data[idx]["text"],
                        "similarity_score": float(similarities[idx])
                    })
            
            return evidence
            
        except Exception as e:
            logging.error(f"Error in TF-IDF positional retrieval: {e}")
            return []
    
    def retrieve_evidence_bm25_positional(self, query: str, chunk_data: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve evidence using BM25 with positional chunks."""
        if not BM25_AVAILABLE:
            raise ImportError("BM25 dependencies not available")
        
        if not chunk_data:
            return []
        
        try:
            # Extract texts for BM25
            chunk_texts = [item["text"] for item in chunk_data]
            
            # Create BM25 index
            bm25 = BM25Okapi([text.split() for text in chunk_texts])
            
            # Get scores
            scores = bm25.get_scores(query.split())
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Return evidence with chunk IDs
            evidence = []
            for idx in top_indices:
                if scores[idx] > 0.1:  # Minimum score threshold
                    evidence.append({
                        "chunk_id": chunk_data[idx]["id"],
                        "text": chunk_data[idx]["text"],
                        "bm25_score": float(scores[idx])
                    })
            
            return evidence
            
        except Exception as e:
            logging.error(f"Error in BM25 positional retrieval: {e}")
            return []
    
    def retrieve_evidence_sbert_positional(self, query: str, chunk_data: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve evidence using SBERT with positional chunks."""
        if not SBERT_AVAILABLE:
            raise ImportError("SBERT dependencies not available")
        
        if not chunk_data:
            return []
        
        try:
            # Initialize SBERT model if not already done
            if not hasattr(self, 'sbert_model') or self.sbert_model is None:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Extract texts for SBERT
            chunk_texts = [item["text"] for item in chunk_data]
            
            # Encode query and chunks
            query_embedding = self.sbert_model.encode([query])
            chunk_embeddings = self.sbert_model.encode(chunk_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return evidence with chunk IDs
            evidence = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    evidence.append({
                        "chunk_id": chunk_data[idx]["id"],
                        "text": chunk_data[idx]["text"],
                        "similarity_score": float(similarities[idx])
                    })
            
            return evidence
            
        except Exception as e:
            logging.error(f"Error in SBERT positional retrieval: {e}")
            return []

def retrieve_evidence_for_claim(claim, chunks, method="auto", top_k=3):
    """
    Main function to retrieve evidence for a single claim.
    
    Args:
        claim (str): Claim text
        chunks (list): List of chunk dictionaries
        method (str): Retrieval method
        top_k (int): Number of top chunks
    
    Returns:
        dict: Dictionary with evidence and actual method used
    """
    retriever = EvidenceRetriever()
    return retriever.retrieve_evidence(claim, chunks, method, top_k)

def retrieve_evidence_for_claims(claims, chunks, method="auto", top_k=3):
    """
    Main function to retrieve evidence for multiple claims.
    
    Args:
        claims (list): List of claims
        chunks (list): List of chunk dictionaries
        method (str): Retrieval method
        top_k (int): Number of top chunks per claim
    
    Returns:
        list: List of claims with evidence
    """
    retriever = EvidenceRetriever()
    return retriever.retrieve_evidence_for_claims(claims, chunks, method, top_k)
