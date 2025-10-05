"""
Main Orchestration Module
Coordinates the entire OpenReview paper processing pipeline
"""

import os
import json
import logging
import argparse
import datetime
import ssl
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

# Import all modules - using the correct module names with numbers
import importlib.util

# Load modules dynamically since they have numbers in their names
def load_module(module_path, function_name=None, class_name=None):
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if function_name:
        return getattr(module, function_name)
    elif class_name:
        return getattr(module, class_name)
    return module

# Load functions and classes
crawl_openreview_url = load_module("app/1_crawl_openreview.py", "crawl_openreview_url")
parse_pdf_to_markdown = load_module("app/2_parse_pdf.py", "parse_pdf_to_markdown")
clean_markdown = load_module("app/3_clean_markdown.py", "clean_markdown")
chunk_markdown = load_module("app/4_chunk_markdown.py", "chunk_markdown")
chunk_positional = load_module("app/4a_chunk_positional.py", "chunk_positional")
extract_reviews_from_metadata = load_module("app/5_extract_structured_reviews.py", "extract_reviews_from_metadata")
extract_claims_from_reviews = load_module("app/6_claim_extraction.py", "extract_claims_from_reviews")
EvidenceRetriever = load_module("app/7_evidence_retrieval.py", class_name="EvidenceRetriever")
ClaimVerifier = load_module("app/8_claim_verification.py", class_name="ClaimVerifier")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

class RunLogger:
    """Enhanced logging class to track all configuration and methods used in each run."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.run_logs_dir = output_dir / "run_logs"
        self.run_logs_dir.mkdir(exist_ok=True)
        
    def create_run_log(self, url: str, config: Dict[str, Any]) -> str:
        """
        Create a comprehensive run log file with all configuration and methods.
        
        Args:
            url (str): OpenReview URL being processed
            config (dict): Configuration dictionary
            
        Returns:
            str: Path to the run log file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        log_file = self.run_logs_dir / f"{run_id}_config.json"
        
        # Extract submission ID from URL
        submission_id = None
        if "forum?id=" in url:
            submission_id = url.split("forum?id=")[-1]
        
        # Create comprehensive run log
        run_log = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "url": url,
            "submission_id": submission_id,
            "configuration": {
                "pdf_parser": {
                    "method": config.get("pdf_parser", "auto"),
                    "parser_kwargs": config.get("parser_kwargs", {}),
                    "description": "Method used to parse PDF to markdown"
                },
                "chunking": {
                    "chunk_size": config.get("chunk_size", 512),
                    "description": "Maximum tokens per chunk for text chunking"
                },
                "claim_extraction": {
                    "method": config.get("claim_extraction", "auto"),
                    "available_methods": ["fenice", "rule_based", "auto"],
                    "description": "Method used to extract claims from reviews"
                },
                "evidence_retrieval": {
                    "method": config.get("evidence_retrieval", "auto"),
                    "top_k": config.get("top_k", 4),
                    "available_methods": ["tfidf", "bm25", "sbert", "auto"],
                    "description": "Method used to retrieve evidence for claims"
                },
                "claim_verification": {
                    "model": config.get("verification_model", "qwen3:8b"),
                    "temperature": 0.0,
                    "num_predict": 2048,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "seed": 42,
                    "description": "LLM model and parameters for claim verification"
                },
                "output_directory": config.get("output_dir", "outputs")
            },
            "actual_methods_used": {
                "pdf_parser": {
                    "configured_method": config.get("pdf_parser", "auto"),
                    "actual_method": None,
                    "fallback_chain": [],
                    "description": "Actual method used after fallback resolution"
                },
                "claim_extraction": {
                    "configured_method": config.get("claim_extraction", "auto"),
                    "actual_method": None,
                    "fallback_chain": [],
                    "description": "Actual method used after fallback resolution"
                },
                "evidence_retrieval": {
                    "configured_method": config.get("evidence_retrieval", "auto"),
                    "actual_method": None,
                    "fallback_chain": [],
                    "description": "Actual method used after fallback resolution"
                },
                "claim_verification": {
                    "configured_model": config.get("verification_model", "qwen3:8b"),
                    "actual_model": None,
                    "description": "Actual model used for verification"
                }
            },
            "pipeline_steps": [
                {
                    "step": 1,
                    "name": "OpenReview Crawling",
                    "method": "crawl_openreview_url",
                    "description": "Crawl OpenReview URL to extract paper and reviews"
                },
                {
                    "step": 2,
                    "name": "PDF Parsing",
                    "method": "parse_pdf_to_markdown",
                    "description": "Convert PDF to markdown format"
                },
                {
                    "step": 3,
                    "name": "Markdown Cleaning",
                    "method": "clean_markdown",
                    "description": "Clean and preprocess markdown text"
                },
                {
                    "step": 4,
                    "name": "Text Chunking",
                    "method": "chunk_markdown",
                    "description": "Split text into manageable chunks"
                },
                {
                    "step": 5,
                    "name": "Review Extraction",
                    "method": "extract_reviews_from_metadata",
                    "description": "Extract structured reviews from metadata"
                },
                {
                    "step": 6,
                    "name": "Claim Extraction",
                    "method": "extract_claims_from_reviews",
                    "description": "Extract claims from review text"
                },
                {
                    "step": 7,
                    "name": "Evidence Retrieval",
                    "method": "EvidenceRetriever.retrieve_evidence",
                    "description": "Retrieve relevant evidence for claims"
                },
                {
                    "step": 8,
                    "name": "Claim Verification",
                    "method": "ClaimVerifier.verify_claims_batch",
                    "description": "Verify claims against evidence using LLM"
                }
            ],
            "system_info": {
                "python_version": os.sys.version,
                "platform": os.sys.platform,
                "working_directory": os.getcwd()
            }
        }
        
        # Save run log
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(run_log, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Created run log: {log_file}")
        return str(log_file)
    
    def update_actual_method_used(self, log_file: str, method_type: str, actual_method: str, fallback_chain: List[str] = None):
        """
        Update the log with the actual method used for a specific component.
        
        Args:
            log_file (str): Path to run log file
            method_type (str): Type of method (pdf_parser, claim_extraction, evidence_retrieval, claim_verification)
            actual_method (str): The actual method that was used
            fallback_chain (list): List of methods tried in order (for fallback scenarios)
        """
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                run_log = json.load(f)
            
            if method_type in run_log.get("actual_methods_used", {}):
                run_log["actual_methods_used"][method_type]["actual_method"] = actual_method
                if fallback_chain:
                    run_log["actual_methods_used"][method_type]["fallback_chain"] = fallback_chain
                
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(run_log, f, indent=2, ensure_ascii=False)
                    
                logging.info(f"Updated actual method for {method_type}: {actual_method}")
                
        except Exception as e:
            logging.error(f"Failed to update actual method for {method_type}: {e}")
    
    def update_run_log(self, log_file: str, step_results: Dict[str, Any]):
        """
        Update run log with step results.
        
        Args:
            log_file (str): Path to run log file
            step_results (dict): Results from processing steps
        """
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                run_log = json.load(f)
            
            # Add step results
            if 'step_results' not in run_log:
                run_log['step_results'] = {}
            
            run_log['step_results'].update(step_results)
            
            # Add completion timestamp
            run_log['completion_timestamp'] = datetime.datetime.now().isoformat()
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(run_log, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Failed to update run log: {e}")

class OpenReviewProcessor:
    def __init__(self, config: Dict[str, Any], progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the OpenReview processor.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.output_dir.mkdir(exist_ok=True)
        self.progress_callback = progress_callback
        
        # Initialize run logger
        self.run_logger = RunLogger(self.output_dir)
        
        # Create subdirectories
        (self.output_dir / "pdfs").mkdir(exist_ok=True)
        (self.output_dir / "markdown").mkdir(exist_ok=True)
        (self.output_dir / "chunks").mkdir(exist_ok=True)
        (self.output_dir / "reviews").mkdir(exist_ok=True)
        (self.output_dir / "claims").mkdir(exist_ok=True)
        (self.output_dir / "evidence").mkdir(exist_ok=True)
        (self.output_dir / "verification").mkdir(exist_ok=True)
        (self.output_dir / "positional").mkdir(exist_ok=True)
        
        logging.info(f"Initialized processor with output directory: {self.output_dir}")
    
    def _emit_progress(self, payload: Dict[str, Any]) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(payload)
            except Exception:
                # Never let progress reporting break the pipeline
                pass
    
    def process_openreview_url(self, url: str, username: str = None, password: str = None) -> Dict[str, Any]:
        """
        Process a single OpenReview URL through the entire pipeline.
        
        Args:
            url (str): OpenReview URL
            username (str): OpenReview username (optional)
            password (str): OpenReview password (optional)
        
        Returns:
            dict: Processing results
        """
        logging.info(f"Starting processing for URL: {url}")
        
        # Create run log
        run_log_file = self.run_logger.create_run_log(url, self.config)
        
        results = {
            'url': url,
            'run_log_file': run_log_file,
            'submission_id': None,
            'pdf_path': None,
            'markdown_path': None,
            'cleaned_markdown_path': None,
            'chunks_path': None,
            'reviews_path': None,
            'claims_path': None,
            'evidence_path': None,
            'verification_path': None,
            'report_path': None,
            'success': False,
            'error': None
        }
        
        step_results = {}
        
        try:
            self._emit_progress({
                'phase': 'start',
                'status': 'running',
                'message': 'Pipeline started'
            })
            # Step 1: Crawl OpenReview URL
            logging.info("Step 1: Crawling OpenReview URL")
            self._emit_progress({'phase': 'crawl', 'status': 'running', 'message': 'Crawling OpenReview', 'progress': 0})
            metadata = crawl_openreview_url(url, str(self.output_dir), username, password)
            results['submission_id'] = metadata.get('submission_id')
            results['pdf_path'] = metadata.get('pdf_local_path')
            
            step_results['step_1_crawling'] = {
                'status': 'completed',
                'submission_id': results['submission_id'],
                'pdf_path': results['pdf_path'],
                'metadata_keys': list(metadata.keys()) if metadata else []
            }
            self._emit_progress({'phase': 'crawl', 'status': 'completed', 'submission_id': results['submission_id']})
            
            if not results['pdf_path']:
                raise ValueError("No PDF found for this submission")
            
            # Step 2: Parse PDF to Markdown
            logging.info("Step 2: Parsing PDF to Markdown")
            self._emit_progress({'phase': 'parse_pdf', 'status': 'running', 'message': 'Parsing PDF to markdown'})
            md_path = self.output_dir / "markdown" / f"{results['submission_id']}.md"
            pdf_result = parse_pdf_to_markdown(
                results['pdf_path'],
                str(md_path),
                method=self.config.get("pdf_parser", "auto"),
                **self.config.get("parser_kwargs", {})
            )
            results['markdown_path'] = pdf_result['output_path']
            
            # Update actual method used in log
            self.run_logger.update_actual_method_used(
                run_log_file, 
                "pdf_parser", 
                pdf_result['actual_method'],
                pdf_result['fallback_chain']
            )
            
            step_results['step_2_pdf_parsing'] = {
                'status': 'completed',
                'configured_method': self.config.get("pdf_parser", "auto"),
                'actual_method': pdf_result['actual_method'],
                'fallback_chain': pdf_result['fallback_chain'],
                'parser_kwargs': self.config.get("parser_kwargs", {}),
                'markdown_path': str(md_path)
            }
            self._emit_progress({'phase': 'parse_pdf', 'status': 'completed', 'actual_method': pdf_result['actual_method']})
            
            # Step 3: Clean Markdown
            logging.info("Step 3: Cleaning Markdown")
            self._emit_progress({'phase': 'clean_markdown', 'status': 'running'})
            cleaned_md_path = self.output_dir / "markdown" / f"{results['submission_id']}_cleaned.md"
            clean_markdown(
                str(md_path),
                str(cleaned_md_path),
                char_thresh=10,
                punct_seq_thresh=10,
                remove_checklists=True,
                remove_figures=True
            )
            results['cleaned_markdown_path'] = str(cleaned_md_path)
            
            step_results['step_3_markdown_cleaning'] = {
                'status': 'completed',
                'cleaned_markdown_path': str(cleaned_md_path)
            }
            self._emit_progress({'phase': 'clean_markdown', 'status': 'completed'})
            
            # Step 4: Chunking
            positional_enabled = bool(self.config.get("positional_chunking", False))
            if positional_enabled:
                logging.info("Step 4: Positional chunking (V5 column-aware)")
                self._emit_progress({'phase': 'chunk', 'status': 'running', 'mode': 'positional_v5'})
                pos_dir = self.output_dir / "positional" / results['submission_id']
                pos_dir.mkdir(parents=True, exist_ok=True)
                pos_result = chunk_positional(
                    results['pdf_path'],
                    str(pos_dir),
                    max_tokens=self.config.get("chunk_size", 512),
                    column_split_x=self.config.get("column_split_x", 300.0)
                )
                results['chunks_path'] = pos_result['chunks_path']
                results['positional_dir'] = str(pos_dir)
                results['docling_dict_path'] = pos_result['docling_dict_path']
                results['docling_markdown_path'] = pos_result['docling_markdown_path']

                step_results['step_4_chunking'] = {
                    'status': 'completed',
                    'mode': 'positional_v5',
                    'chunk_size': self.config.get("chunk_size", 512),
                    'column_split_x': self.config.get("column_split_x", 300.0),
                    'chunks_path': pos_result['chunks_path']
                }
                try:
                    import os
                    num_chunks = 0
                    if os.path.exists(pos_result['chunks_path']):
                        with open(pos_result['chunks_path'], 'r', encoding='utf-8') as _f:
                            _chunks = json.load(_f)
                            if isinstance(_chunks, list):
                                num_chunks = len(_chunks)
                    self._emit_progress({'phase': 'chunk', 'status': 'completed', 'num_chunks': num_chunks})
                except Exception:
                    self._emit_progress({'phase': 'chunk', 'status': 'completed'})
            else:
                logging.info("Step 4: Chunking Markdown (simple)")
                self._emit_progress({'phase': 'chunk', 'status': 'running', 'mode': 'markdown_simple'})
                chunks_path = self.output_dir / "chunks" / f"{results['submission_id']}_chunks.jsonl"
                chunk_markdown(
                    str(cleaned_md_path),
                    str(chunks_path),
                    max_tokens=self.config.get("chunk_size", 512)
                )
                results['chunks_path'] = str(chunks_path)

                step_results['step_4_chunking'] = {
                    'status': 'completed',
                    'mode': 'markdown_simple',
                    'chunk_size': self.config.get("chunk_size", 512),
                    'chunks_path': str(chunks_path)
                }
                self._emit_progress({'phase': 'chunk', 'status': 'completed'})
            
            # Step 5: Extract Structured Reviews
            logging.info("Step 5: Extracting Structured Reviews")
            self._emit_progress({'phase': 'reviews', 'status': 'running'})
            metadata_path = self.output_dir / "pdfs" / f"{results['submission_id']}_metadata.json"
            reviews_path = self.output_dir / "reviews" / f"{results['submission_id']}_reviews.json"
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Try multiple approaches to load reviews
            reviews = []
            
            # First, try to extract from metadata
            if metadata.get('reviews'):
                reviews = extract_reviews_from_metadata(metadata)
                logging.info(f"Extracted {len(reviews)} reviews from metadata")
            
            # If no reviews in metadata, try loading from individual review files
            if not reviews and metadata.get('individual_review_files'):
                logging.info("No reviews in metadata, trying individual review files")
                from pathlib import Path
                extractor_class = load_module("app/5_extract_structured_reviews.py", class_name="ReviewExtractor")
                extractor = extractor_class()
                reviews = extractor.load_individual_review_files(
                    str(self.output_dir / "reviews"), 
                    results['submission_id']
                )
            
            # If still no reviews, try loading from all_reviews.json
            if not reviews and metadata.get('all_reviews_path'):
                logging.info("Trying to load from all_reviews.json")
                extractor_class = load_module("app/5_extract_structured_reviews.py", class_name="ReviewExtractor")
                extractor = extractor_class()
                all_reviews = extractor.load_reviews_from_json(metadata['all_reviews_path'])
                if all_reviews:
                    # Convert to structured format
                    fake_metadata = {
                        'submission_id': results['submission_id'],
                        'title': metadata.get('title'),
                        'reviews': all_reviews
                    }
                    reviews = extractor.extract_reviews_from_metadata(fake_metadata)
            
            # If still no reviews, try loading from pickle backup
            if not reviews and metadata.get('all_reviews_pickle_path'):
                logging.info("Trying to load from pickle backup")
                extractor_class = load_module("app/5_extract_structured_reviews.py", class_name="ReviewExtractor")
                extractor = extractor_class()
                all_reviews = extractor.load_reviews_from_pickle(metadata['all_reviews_pickle_path'])
                if all_reviews:
                    # Convert to structured format
                    fake_metadata = {
                        'submission_id': results['submission_id'],
                        'title': metadata.get('title'),
                        'reviews': all_reviews
                    }
                    reviews = extractor.extract_reviews_from_metadata(fake_metadata)
            
            # Save structured reviews
            if reviews:
                with open(reviews_path, 'w', encoding='utf-8') as f:
                    json.dump(reviews, f, indent=2, ensure_ascii=False)
                logging.info(f"Saved {len(reviews)} structured reviews to {reviews_path}")
            else:
                logging.warning("No reviews found for this submission")
                # Create empty reviews file
                with open(reviews_path, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            
            results['reviews_path'] = str(reviews_path)
            
            step_results['step_5_review_extraction'] = {
                'status': 'completed',
                'reviews_path': str(reviews_path),
                'num_reviews': len(reviews),
                'review_ids': [r.get('review_id') for r in reviews] if reviews else []
            }
            self._emit_progress({'phase': 'reviews', 'status': 'completed', 'num_reviews': len(reviews)})
            
            if not reviews:
                logging.warning("No reviews found for this submission - skipping remaining steps")
                step_results['step_6_claim_extraction'] = {'status': 'skipped', 'reason': 'no_reviews'}
                step_results['step_7_evidence_retrieval'] = {'status': 'skipped', 'reason': 'no_reviews'}
                step_results['step_8_claim_verification'] = {'status': 'skipped', 'reason': 'no_reviews'}
                self.run_logger.update_run_log(run_log_file, step_results)
                results['success'] = True
                self._emit_progress({'phase': 'done', 'status': 'completed', 'message': 'No reviews found'})
                return results
            
            # Step 6: Extract Claims from Reviews
            logging.info("Step 6: Extracting Claims from Reviews")
            self._emit_progress({'phase': 'claims', 'status': 'running'})
            claims_path = self.output_dir / "claims" / f"{results['submission_id']}_claims.json"
            claims_result = extract_claims_from_reviews(
                reviews,
                method=self.config.get("claim_extraction", "auto")
            )
            reviews_with_claims = claims_result['reviews_with_claims']
            
            # Save claims
            with open(claims_path, 'w', encoding='utf-8') as f:
                json.dump(reviews_with_claims, f, indent=2, ensure_ascii=False)
            results['claims_path'] = str(claims_path)
            
            # Update actual method used in log
            self.run_logger.update_actual_method_used(
                run_log_file, 
                "claim_extraction", 
                claims_result['actual_method'],
                claims_result['fallback_chain']
            )
            
            # Collect all claims
            all_claims = []
            for review in reviews_with_claims:
                claims = review.get('extracted_claims', [])
                for claim in claims:
                    all_claims.append({
                        'claim': claim,
                        'review_id': review.get('review_id'),
                        'reviewer': review.get('reviewer')
                    })
            self._emit_progress({'phase': 'claims', 'status': 'completed', 'num_claims': len(all_claims)})
            
            step_results['step_6_claim_extraction'] = {
                'status': 'completed',
                'configured_method': self.config.get("claim_extraction", "auto"),
                'actual_method': claims_result['actual_method'],
                'fallback_chain': claims_result['fallback_chain'],
                'claims_path': str(claims_path),
                'num_claims': len(all_claims),
                'claims_per_review': {r.get('review_id'): len(r.get('extracted_claims', [])) for r in reviews_with_claims}
            }
            
            if not all_claims:
                logging.warning("No claims extracted from reviews")
                step_results['step_7_evidence_retrieval'] = {'status': 'skipped', 'reason': 'no_claims'}
                step_results['step_8_claim_verification'] = {'status': 'skipped', 'reason': 'no_claims'}
                self.run_logger.update_run_log(run_log_file, step_results)
                results['success'] = True
                self._emit_progress({'phase': 'done', 'status': 'completed', 'message': 'No claims found'})
                return results
            
            # Step 7: Retrieve Evidence for Claims
            logging.info("Step 7: Retrieving Evidence for Claims")
            self._emit_progress({'phase': 'evidence', 'status': 'running', 'total_claims': len(all_claims), 'current': 0})
            evidence_path = self.output_dir / "evidence" / f"{results['submission_id']}_evidence.json"
            
            # Load chunks - handle both positional and markdown chunking modes
            retriever = EvidenceRetriever()
            if positional_enabled:
                # Load positional chunks directly (no conversion needed)
                with open(results['chunks_path'], 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
            else:
                # Load markdown chunks
                chunks = retriever.load_chunks_from_jsonl(str(results['chunks_path']))
            
            # Retrieve evidence for each claim
            claims_with_evidence = []
            actual_evidence_method = None
            evidence_fallback_chain = []
            
            for i, claim_data in enumerate(all_claims):
                evidence_result = retriever.retrieve_evidence(
                    claim_data['claim'],
                    chunks,
                    method=self.config.get("evidence_retrieval", "auto"),
                    top_k=self.config.get("top_k", 4)
                )
                
                evidence = evidence_result['evidence']
                actual_evidence_method = evidence_result['actual_method']
                evidence_fallback_chain = evidence_result['fallback_chain']
                
                claims_with_evidence.append({
                    'claim_id': i + 1,
                    'claim': claim_data['claim'],
                    'evidence': evidence,
                    'review_id': claim_data['review_id'],
                    'reviewer': claim_data['reviewer']
                })
                self._emit_progress({'phase': 'evidence', 'status': 'running', 'total_claims': len(all_claims), 'current': i + 1})
            
            # Update actual method used in log
            self.run_logger.update_actual_method_used(
                run_log_file, 
                "evidence_retrieval", 
                actual_evidence_method,
                evidence_fallback_chain
            )
            
            # Save evidence results
            with open(evidence_path, 'w', encoding='utf-8') as f:
                json.dump(claims_with_evidence, f, indent=2, ensure_ascii=False)
            results['evidence_path'] = str(evidence_path)
            
            step_results['step_7_evidence_retrieval'] = {
                'status': 'completed',
                'configured_method': self.config.get("evidence_retrieval", "auto"),
                'actual_method': actual_evidence_method,
                'fallback_chain': evidence_fallback_chain,
                'top_k': self.config.get("top_k", 4),
                'evidence_path': str(evidence_path),
                'num_chunks': len(chunks),
                'num_claims_with_evidence': len(claims_with_evidence)
            }
            self._emit_progress({'phase': 'evidence', 'status': 'completed', 'num_claims': len(claims_with_evidence)})
            
            # Step 8: Verify Claims
            logging.info("Step 8: Verifying Claims")
            self._emit_progress({'phase': 'verify', 'status': 'running', 'total_claims': len(claims_with_evidence), 'current': 0})
            verification_path = self.output_dir / "verification" / f"{results['submission_id']}_verification.json"
            report_path = self.output_dir / "verification" / f"{results['submission_id']}_report.md"
            
            verification_model = self.config.get("verification_model", "qwen3:8b")
            
            verifier = ClaimVerifier()
            # Wrap verification to emit progress after each claim
            verification_results: List[Dict[str, Any]] = []
            for idx, claim_payload in enumerate(claims_with_evidence):
                single_result = verifier.verify_claims_batch([claim_payload], model=verification_model, delay=0.0)
                if single_result:
                    verification_results.append(single_result[0])
                self._emit_progress({'phase': 'verify', 'status': 'running', 'total_claims': len(claims_with_evidence), 'current': idx + 1})
            
            # Update actual method used in log
            self.run_logger.update_actual_method_used(
                run_log_file, 
                "claim_verification", 
                verification_model,
                None  # No fallback chain for verification model
            )
            
            # Save verification results
            verifier.save_verification_results(verification_results, str(verification_path))
            verifier.generate_verification_report(verification_results, str(report_path))
            
            results['verification_path'] = str(verification_path)
            results['report_path'] = str(report_path)
            
            step_results['step_8_claim_verification'] = {
                'status': 'completed',
                'configured_model': self.config.get("verification_model", "qwen3:8b"),
                'actual_model': verification_model,
                'temperature': 0.0,
                'verification_path': str(verification_path),
                'report_path': str(report_path),
                'num_claims_verified': len(verification_results) if verification_results else 0
            }
            
            results['success'] = True
            
            # Update run log with all step results
            self.run_logger.update_run_log(run_log_file, step_results)
            
            logging.info(f"Successfully processed submission {results['submission_id']}")
            self._emit_progress({'phase': 'done', 'status': 'completed', 'submission_id': results['submission_id']})
            
        except Exception as e:
            logging.error(f"Error processing URL {url}: {e}")
            results['error'] = str(e)
            results['success'] = False
            
            # Update run log with error
            step_results['error'] = {
                'timestamp': datetime.datetime.now().isoformat(),
                'error_message': str(e),
                'error_type': type(e).__name__
            }
            self.run_logger.update_run_log(run_log_file, step_results)
            self._emit_progress({'phase': 'done', 'status': 'failed', 'error': str(e)})
        
        return results

def main():
    """Main function to run the OpenReview processor."""
    parser = argparse.ArgumentParser(description="Process OpenReview papers and reviews")
    parser.add_argument("url", help="OpenReview URL to process")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--username", help="OpenReview username")
    parser.add_argument("--password", help="OpenReview password")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "pdf_parser": "auto",
            "parser_kwargs": {},
            "chunk_size": 512,
            "positional_chunking": True,
            "column_split_x": 300.0,
            "claim_extraction": "auto",
            "evidence_retrieval": "auto",
            "verification_model": "qwen3:8b",
            "top_k": 4,
            "output_dir": "outputs"
        }
    
    # Create processor and run
    processor = OpenReviewProcessor(config)
    results = processor.process_openreview_url(args.url, args.username, args.password)
    
    # Print results
    if results['success']:
        print(f"\n✅ Successfully processed submission {results['submission_id']}")
        print(f"📋 Run Log: {results['run_log_file']}")
        print(f"📄 PDF: {results['pdf_path']}")
        print(f"📝 Markdown: {results['markdown_path']}")
        print(f"🧹 Cleaned Markdown: {results['cleaned_markdown_path']}")
        print(f"📦 Chunks: {results['chunks_path']}")
        if results.get('positional_dir'):
            print(f"📐 Positional artifacts: {results['positional_dir']}")
        print(f"📋 Reviews: {results['reviews_path']}")
        print(f"🎯 Claims: {results['claims_path']}")
        print(f"🔍 Evidence: {results['evidence_path']}")
        print(f"✅ Verification: {results['verification_path']}")
        print(f"📊 Report: {results['report_path']}")
        
        # Display configuration summary
        print(f"\n🔧 Configuration Summary:")
        print(f"   • PDF Parser: {config.get('pdf_parser', 'auto')}")
        print(f"   • Positional Chunking: {config.get('positional_chunking', False)}")
        if config.get('positional_chunking', False):
            print(f"   • Column Split X: {config.get('column_split_x', 300.0)}")
        print(f"   • Claim Extraction: {config.get('claim_extraction', 'auto')}")
        print(f"   • Evidence Retrieval: {config.get('evidence_retrieval', 'auto')}")
        print(f"   • Verification Model: {config.get('verification_model', 'qwen3:8b')}")
        print(f"   • Chunk Size: {config.get('chunk_size', 512)}")
        print(f"   • Top-K Evidence: {config.get('top_k', 4)}")
        
        # Display actual methods used (if available)
        try:
            with open(results['run_log_file'], 'r', encoding='utf-8') as f:
                run_log = json.load(f)
            
            actual_methods = run_log.get('actual_methods_used', {})
            if any(method.get('actual_method') for method in actual_methods.values()):
                print(f"\n🔍 Actual Methods Used:")
                if actual_methods.get('pdf_parser', {}).get('actual_method'):
                    print(f"   • PDF Parser: {actual_methods['pdf_parser']['actual_method']}")
                if actual_methods.get('claim_extraction', {}).get('actual_method'):
                    print(f"   • Claim Extraction: {actual_methods['claim_extraction']['actual_method']}")
                if actual_methods.get('evidence_retrieval', {}).get('actual_method'):
                    print(f"   • Evidence Retrieval: {actual_methods['evidence_retrieval']['actual_method']}")
                if actual_methods.get('claim_verification', {}).get('actual_model'):
                    print(f"   • Verification Model: {actual_methods['claim_verification']['actual_model']}")
        except Exception as e:
            logging.warning(f"Could not display actual methods used: {e}")
    else:
        print(f"\n❌ Failed to process URL: {results['error']}")
        if 'run_log_file' in results:
            print(f"📋 Run Log: {results['run_log_file']}")

if __name__ == "__main__":
    main()
