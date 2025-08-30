"""
Claim Extraction Module
Extracts claims from review text using multiple methods
"""

import re
import logging
from typing import List, Dict, Any

# Try to import ML dependencies
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    FENICE_AVAILABLE = True
except ImportError:
    FENICE_AVAILABLE = False
    logging.warning("FENICE dependencies not available. Using rule-based extraction.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import nltk
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False
    logging.warning("Gemma dependencies not available.")

class ClaimExtractor:
    def __init__(self):
        """Initialize claim extractor."""
        self.fenice_model = None
        self.fenice_tokenizer = None
        self.gemma_pipeline = None
        
        # Initialize models if available
        if FENICE_AVAILABLE:
            self._load_fenice_model()
        
        if GEMMA_AVAILABLE:
            self._load_gemma_model()
    
    def _load_fenice_model(self):
        """Load FENICE model for claim extraction."""
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            
            self.fenice_tokenizer = T5Tokenizer.from_pretrained("Babelscape/t5-base-summarization-claim-extractor")
            self.fenice_model = T5ForConditionalGeneration.from_pretrained("Babelscape/t5-base-summarization-claim-extractor")
            self.fenice_model.to(device)
            logging.info("FENICE model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load FENICE model: {e}")
            FENICE_AVAILABLE = False
    
    def _load_gemma_model(self):
        """Load Gemma model for claim extraction."""
        try:
            # This is a placeholder for Gemma model loading
            # You would need to implement the actual Gemma claim extraction
            logging.info("Gemma model loading not implemented yet")
        except Exception as e:
            logging.error(f"Failed to load Gemma model: {e}")
            GEMMA_AVAILABLE = False
    
    def extract_claims_fenice(self, review_text: str) -> List[str]:
        """
        Extract claims using FENICE model.
        
        Args:
            review_text (str): Review text to extract claims from
        
        Returns:
            list: List of extracted claims
        """
        if not FENICE_AVAILABLE or self.fenice_model is None:
            raise ImportError("FENICE model not available")
        
        try:
            # Prepare input for the model
            tok_input = self.fenice_tokenizer.batch_encode_plus([review_text], return_tensors="pt", padding=True)
            
            if torch.cuda.is_available():
                tok_input = {key: value.cuda() for key, value in tok_input.items()}
            
            # Generate claims
            claims = self.fenice_model.generate(**tok_input)
            
            # Decode the generated claims
            claims = self.fenice_tokenizer.batch_decode(claims, skip_special_tokens=True)
            
            # Split claims into list and clean up
            claims_list = [s.strip() for s in re.split(r'(?<=[.!?])\s+', claims[0]) if s.strip()]
            
            # Remove duplicates while preserving order
            claims_list = list(dict.fromkeys(claims_list))
            
            return claims_list
            
        except Exception as e:
            logging.error(f"Error in FENICE claim extraction: {e}")
            return []
    
    def extract_claims_gemma(self, review_text: str) -> List[str]:
        """
        Extract claims using Gemma model.
        
        Args:
            review_text (str): Review text to extract claims from
        
        Returns:
            list: List of extracted claims
        """
        if not GEMMA_AVAILABLE:
            raise ImportError("Gemma model not available")
        
        # Placeholder implementation
        # You would need to implement the actual Gemma claim extraction logic
        logging.warning("Gemma claim extraction not implemented yet")
        return []
    
    def extract_claims_rule_based(self, review_text: str) -> List[str]:
        """
        Extract claims using rule-based approach.
        
        Args:
            review_text (str): Review text to extract claims from
        
        Returns:
            list: List of extracted claims
        """
        claims = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', review_text)
        
        # Keywords that often indicate claims
        claim_keywords = [
            'propose', 'proposes', 'proposed', 'proposing',
            'introduce', 'introduces', 'introduced', 'introducing',
            'present', 'presents', 'presented', 'presenting',
            'demonstrate', 'demonstrates', 'demonstrated', 'demonstrating',
            'show', 'shows', 'showed', 'showing',
            'achieve', 'achieves', 'achieved', 'achieving',
            'improve', 'improves', 'improved', 'improving',
            'outperform', 'outperforms', 'outperformed', 'outperforming',
            'better', 'best', 'superior', 'excellent',
            'novel', 'new', 'original', 'innovative',
            'effective', 'efficient', 'robust', 'scalable',
            'limitation', 'limitations', 'weakness', 'weaknesses',
            'problem', 'problems', 'issue', 'issues',
            'fail', 'fails', 'failed', 'failing',
            'lack', 'lacks', 'missing', 'absent'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Check if sentence contains claim keywords
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in claim_keywords):
                claims.append(sentence)
        
        return claims
    
    def extract_claims(self, review_text: str, method: str = "auto") -> List[str]:
        """
        Extract claims from review text using specified method.
        
        Args:
            review_text (str): Review text to extract claims from
            method (str): Extraction method ("auto", "fenice", "gemma", "rule_based")
        
        Returns:
            list: List of extracted claims
        """
        if method == "auto":
            # Try methods in order of preference
            if FENICE_AVAILABLE and self.fenice_model is not None:
                method = "fenice"
            elif GEMMA_AVAILABLE and self.gemma_pipeline is not None:
                method = "gemma"
            else:
                method = "rule_based"
        
        logging.info(f"Extracting claims using method: {method}")
        
        if method == "fenice":
            return self.extract_claims_fenice(review_text)
        elif method == "gemma":
            return self.extract_claims_gemma(review_text)
        elif method == "rule_based":
            return self.extract_claims_rule_based(review_text)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def extract_claims_batch(self, reviews: List[Dict[str, Any]], method: str = "auto") -> List[Dict[str, Any]]:
        """
        Extract claims from multiple reviews.
        
        Args:
            reviews (list): List of review dictionaries
            method (str): Extraction method
        
        Returns:
            list: List of reviews with extracted claims
        """
        results = []
        
        for review in reviews:
            review_text = review.get('full_review', '')
            if not review_text:
                # Fallback to individual sections
                sections = []
                if review.get('review_text'):
                    sections.append(review['review_text'])
                if review.get('strengths'):
                    sections.append(f"Strengths: {review['strengths']}")
                if review.get('weaknesses'):
                    sections.append(f"Weaknesses: {review['weaknesses']}")
                review_text = '\n\n'.join(sections)
            
            claims = self.extract_claims(review_text, method)
            
            # Add claims to review
            review_with_claims = review.copy()
            review_with_claims['extracted_claims'] = claims
            results.append(review_with_claims)
        
        return results
    
    def save_claims(self, reviews_with_claims: List[Dict[str, Any]], output_path: str):
        """
        Save reviews with extracted claims to JSON file.
        
        Args:
            reviews_with_claims (list): List of reviews with claims
            output_path (str): Path to output JSON file
        """
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reviews_with_claims, f, indent=2, ensure_ascii=False)
        
        total_claims = sum(len(review.get('extracted_claims', [])) for review in reviews_with_claims)
        logging.info(f"Saved {len(reviews_with_claims)} reviews with {total_claims} claims to {output_path}")

def extract_claims_from_review(review_text, method="auto"):
    """
    Main function to extract claims from a single review.
    
    Args:
        review_text (str): Review text
        method (str): Extraction method
    
    Returns:
        list: List of extracted claims
    """
    extractor = ClaimExtractor()
    return extractor.extract_claims(review_text, method)

def extract_claims_from_reviews(reviews, method="auto"):
    """
    Main function to extract claims from multiple reviews.
    
    Args:
        reviews (list): List of review dictionaries
        method (str): Extraction method
    
    Returns:
        list: List of reviews with extracted claims
    """
    extractor = ClaimExtractor()
    return extractor.extract_claims_batch(reviews, method)
