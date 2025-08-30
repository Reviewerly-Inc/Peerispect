"""
Claim Verification Module
Verifies claims against evidence using Ollama LLM
"""

import json
import logging
import re
import time
from typing import List, Dict, Any

# Try to import Ollama
try:
    from ollama import chat
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available for claim verification.")

class ClaimVerifier:
    def __init__(self):
        """Initialize claim verifier."""
        if not OLLAMA_AVAILABLE:
            logging.warning("Ollama not available. Claim verification will not work.")
    
    def verify_claim_with_ollama(self, claim: str, evidence: List[str], model: str = "qwen3:14b") -> Dict[str, Any]:
        """
        Verify a claim against evidence using Ollama.
        
        Args:
            claim (str): Claim to verify
            evidence (list): List of evidence texts
            model (str): Ollama model name
        
        Returns:
            dict: Verification result
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama not available")
        
        prompt = self._create_verification_prompt(claim, evidence)
        
        try:
            # Ollama options for inference
            options = {
                'temperature': 0.0,
                'num_predict': 2048,
                'top_k': 40,
                'top_p': 0.9,
                'repeat_penalty': 1.1,
                'seed': 42,
            }
            
            response = chat(model, messages=[{'role': 'user', 'content': prompt}], options=options)
            output = response['message']['content']
            
            # Parse JSON response
            result = self._parse_verification_response(output)
            return result
            
        except Exception as e:
            logging.error(f"Error in Ollama verification: {e}")
            return {
                'result': 'Error',
                'justification': f'Verification failed: {str(e)}',
                'confidence': 0.0
            }
    
    def _create_verification_prompt(self, claim: str, evidence: List[str]) -> str:
        """
        Create verification prompt for Ollama.
        
        Args:
            claim (str): Claim to verify
            evidence (list): List of evidence texts
        
        Returns:
            str: Formatted prompt
        """
        evidence_text = "\n---\n".join(evidence)
        
        prompt = f"""You are a factual verification API. Respond ONLY with a valid JSON object—no other text.

Label definitions you must use:
  • Supported: The evidence's content fully backs the claim with no gaps or contradictions.
  • Partially Supported: Some parts of the claim align with the evidence, but other details are missing or unclear.
  • Contradicted: The claim directly conflicts with the evidence or established facts.
  • Undetermined: The evidence is insufficient to confirm or deny the claim.

Task: Classify the claim using exactly one of the four labels above.

CLAIM:
{claim}

EVIDENCE SNIPPETS (each separated by '---'):
{evidence_text}

CRITICAL OUTPUT RULES:
  • Output ONLY the JSON object—no thinking notes, no markdown, no extra text.
  • The very first character you emit must be '{{' and the very last one must be '}}'.
  • Provide exactly three keys, lowercase, in this order: result, justification, confidence.
  • For "result", choose one of: Supported, Partially Supported, Contradicted, Undetermined.
  • "justification" must be a single concise sentence (≤ 30 words) explaining why the label was chosen.
  • "confidence" must be a number between 0.0 and 1.0 indicating your confidence in the result.
  • Do NOT include additional keys, formatting, or commentary.

Return exactly:
{{
  "result": "<Supported|Partially Supported|Contradicted|Undetermined>",
  "justification": "<brief explanation>",
  "confidence": <0.0-1.0>
}}"""
        
        return prompt
    
    def _parse_verification_response(self, output: str) -> Dict[str, Any]:
        """
        Parse verification response from Ollama.
        
        Args:
            output (str): Raw output from Ollama
        
        Returns:
            dict: Parsed verification result
        """
        try:
            # Try to find JSON in the output
            json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.S))
            if not json_matches:
                raise ValueError("No JSON object found in LLM output.")
            
            # Take the last match (most likely the actual response)
            json_str = json_matches[-1].group()
            
            # Parse JSON
            result = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['result', 'justification', 'confidence']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate result values
            valid_results = ['Supported', 'Partially Supported', 'Contradicted', 'Undetermined']
            if result['result'] not in valid_results:
                raise ValueError(f"Invalid result: {result['result']}")
            
            # Validate confidence
            confidence = float(result['confidence'])
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Invalid confidence: {confidence}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error parsing verification response: {e}")
            return {
                'result': 'Error',
                'justification': f'Failed to parse response: {str(e)}',
                'confidence': 0.0
            }
    
    def verify_claims_batch(self, claims_with_evidence: List[Dict[str, Any]], 
                          model: str = "qwen3:14b", delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Verify multiple claims with evidence.
        
        Args:
            claims_with_evidence (list): List of claims with evidence
            model (str): Ollama model name
            delay (float): Delay between requests in seconds
        
        Returns:
            list: List of verification results
        """
        results = []
        
        for i, claim_data in enumerate(claims_with_evidence):
            claim = claim_data.get('claim', '')
            evidence = claim_data.get('evidence', [])
            
            logging.info(f"Verifying claim {i+1}/{len(claims_with_evidence)}")
            
            verification_result = self.verify_claim_with_ollama(claim, evidence, model)
            
            # Combine claim data with verification result
            result = {
                'claim_id': claim_data.get('claim_id', i+1),
                'claim': claim,
                'evidence': evidence,
                'verification': verification_result,
                'review_id': claim_data.get('review_id', ''),
                'reviewer': claim_data.get('reviewer', '')
            }
            results.append(result)
            
            # Add delay between requests
            if i < len(claims_with_evidence) - 1:
                time.sleep(delay)
        
        return results
    
    def save_verification_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save verification results to JSON file.
        
        Args:
            results (list): List of verification results
            output_path (str): Path to output JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Calculate statistics
        total_claims = len(results)
        supported = sum(1 for r in results if r['verification']['result'] == 'Supported')
        partially_supported = sum(1 for r in results if r['verification']['result'] == 'Partially Supported')
        contradicted = sum(1 for r in results if r['verification']['result'] == 'Contradicted')
        undetermined = sum(1 for r in results if r['verification']['result'] == 'Undetermined')
        
        logging.info(f"Saved verification results to {output_path}")
        logging.info(f"Statistics: {total_claims} total claims")
        logging.info(f"  Supported: {supported}")
        logging.info(f"  Partially Supported: {partially_supported}")
        logging.info(f"  Contradicted: {contradicted}")
        logging.info(f"  Undetermined: {undetermined}")
    
    def generate_verification_report(self, results: List[Dict[str, Any]], output_path: str):
        """
        Generate a human-readable verification report.
        
        Args:
            results (list): List of verification results
            output_path (str): Path to output report file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Claim Verification Report\n\n")
            
            # Summary statistics
            total_claims = len(results)
            supported = sum(1 for r in results if r['verification']['result'] == 'Supported')
            partially_supported = sum(1 for r in results if r['verification']['result'] == 'Partially Supported')
            contradicted = sum(1 for r in results if r['verification']['result'] == 'Contradicted')
            undetermined = sum(1 for r in results if r['verification']['result'] == 'Undetermined')
            
            f.write("## Summary\n\n")
            f.write(f"- Total Claims: {total_claims}\n")
            f.write(f"- Supported: {supported} ({supported/total_claims*100:.1f}%)\n")
            f.write(f"- Partially Supported: {partially_supported} ({partially_supported/total_claims*100:.1f}%)\n")
            f.write(f"- Contradicted: {contradicted} ({contradicted/total_claims*100:.1f}%)\n")
            f.write(f"- Undetermined: {undetermined} ({undetermined/total_claims*100:.1f}%)\n\n")
            
            # Group claims by reviewer
            reviewer_stats = {}
            for result in results:
                reviewer = result.get('reviewer', 'Unknown')
                if reviewer not in reviewer_stats:
                    reviewer_stats[reviewer] = {'total': 0, 'supported': 0, 'partially_supported': 0, 'contradicted': 0, 'undetermined': 0}
                
                reviewer_stats[reviewer]['total'] += 1
                result_type = result['verification']['result']
                if result_type == 'Supported':
                    reviewer_stats[reviewer]['supported'] += 1
                elif result_type == 'Partially Supported':
                    reviewer_stats[reviewer]['partially_supported'] += 1
                elif result_type == 'Contradicted':
                    reviewer_stats[reviewer]['contradicted'] += 1
                elif result_type == 'Undetermined':
                    reviewer_stats[reviewer]['undetermined'] += 1
            
            f.write("## Claims by Reviewer\n\n")
            for reviewer, stats in reviewer_stats.items():
                f.write(f"### {reviewer}\n\n")
                f.write(f"- Total Claims: {stats['total']}\n")
                f.write(f"- Supported: {stats['supported']} ({stats['supported']/stats['total']*100:.1f}%)\n")
                f.write(f"- Partially Supported: {stats['partially_supported']} ({stats['partially_supported']/stats['total']*100:.1f}%)\n")
                f.write(f"- Contradicted: {stats['contradicted']} ({stats['contradicted']/stats['total']*100:.1f}%)\n")
                f.write(f"- Undetermined: {stats['undetermined']} ({stats['undetermined']/stats['total']*100:.1f}%)\n\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            
            for result in results:
                f.write(f"### Claim {result['claim_id']}\n\n")
                f.write(f"**Claim:** {result['claim']}\n\n")
                f.write(f"**Reviewer:** {result.get('reviewer', 'Unknown')} (ID: {result.get('review_id', 'Unknown')})\n\n")
                f.write(f"**Result:** {result['verification']['result']}\n")
                f.write(f"**Confidence:** {result['verification']['confidence']:.2f}\n")
                f.write(f"**Justification:** {result['verification']['justification']}\n\n")
                
                f.write("**Evidence:**\n")
                for i, evidence in enumerate(result['evidence'], 1):
                    f.write(f"{i}. {evidence[:200]}...\n")
                f.write("\n" + "-"*50 + "\n\n")
        
        logging.info(f"Generated verification report: {output_path}")

def verify_claim(claim, evidence, model="qwen3:14b"):
    """
    Main function to verify a single claim.
    
    Args:
        claim (str): Claim text
        evidence (list): List of evidence texts
        model (str): Ollama model name
    
    Returns:
        dict: Verification result
    """
    verifier = ClaimVerifier()
    return verifier.verify_claim_with_ollama(claim, evidence, model)

def verify_claims(claims_with_evidence, model="qwen3:14b", delay=1.0):
    """
    Main function to verify multiple claims.
    
    Args:
        claims_with_evidence (list): List of claims with evidence
        model (str): Ollama model name
        delay (float): Delay between requests
    
    Returns:
        list: List of verification results
    """
    verifier = ClaimVerifier()
    return verifier.verify_claims_batch(claims_with_evidence, model, delay)
