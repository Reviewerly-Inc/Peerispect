"""
Main Orchestration Module
Coordinates the entire OpenReview paper processing pipeline
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

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
crawl_openreview_url = load_module("1_crawl_openreview.py", "crawl_openreview_url")
parse_pdf_to_markdown = load_module("2_parse_pdf.py", "parse_pdf_to_markdown")
clean_markdown = load_module("3_clean_markdown.py", "clean_markdown")
chunk_markdown = load_module("4_chunk_markdown.py", "chunk_markdown")
extract_reviews_from_metadata = load_module("5_extract_structured_reviews.py", "extract_reviews_from_metadata")
extract_claims_from_reviews = load_module("6_claim_extraction.py", "extract_claims_from_reviews")
EvidenceRetriever = load_module("7_evidence_retrieval.py", class_name="EvidenceRetriever")
ClaimVerifier = load_module("8_claim_verification.py", class_name="ClaimVerifier")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

class OpenReviewProcessor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenReview processor.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "pdfs").mkdir(exist_ok=True)
        (self.output_dir / "markdown").mkdir(exist_ok=True)
        (self.output_dir / "chunks").mkdir(exist_ok=True)
        (self.output_dir / "reviews").mkdir(exist_ok=True)
        (self.output_dir / "claims").mkdir(exist_ok=True)
        (self.output_dir / "evidence").mkdir(exist_ok=True)
        (self.output_dir / "verification").mkdir(exist_ok=True)
        
        logging.info(f"Initialized processor with output directory: {self.output_dir}")
    
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
        
        results = {
            'url': url,
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
        
        try:
            # Step 1: Crawl OpenReview URL
            logging.info("Step 1: Crawling OpenReview URL")
            metadata = crawl_openreview_url(url, str(self.output_dir), username, password)
            results['submission_id'] = metadata.get('submission_id')
            results['pdf_path'] = metadata.get('pdf_local_path')
            
            if not results['pdf_path']:
                raise ValueError("No PDF found for this submission")
            
            # Step 2: Parse PDF to Markdown
            logging.info("Step 2: Parsing PDF to Markdown")
            md_path = self.output_dir / "markdown" / f"{results['submission_id']}.md"
            parse_pdf_to_markdown(
                results['pdf_path'],
                str(md_path),
                method=self.config.get("pdf_parser", "auto"),
                **self.config.get("parser_kwargs", {})
            )
            results['markdown_path'] = str(md_path)
            
            # Step 3: Clean Markdown
            logging.info("Step 3: Cleaning Markdown")
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
            
            # Step 4: Chunk Markdown
            logging.info("Step 4: Chunking Markdown")
            chunks_path = self.output_dir / "chunks" / f"{results['submission_id']}_chunks.jsonl"
            chunk_markdown(
                str(cleaned_md_path),
                str(chunks_path),
                max_tokens=self.config.get("chunk_size", 512)
            )
            results['chunks_path'] = str(chunks_path)
            
            # Step 5: Extract Structured Reviews
            logging.info("Step 5: Extracting Structured Reviews")
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
                extractor_class = load_module("5_extract_structured_reviews.py", class_name="ReviewExtractor")
                extractor = extractor_class()
                reviews = extractor.load_individual_review_files(
                    str(self.output_dir / "reviews"), 
                    results['submission_id']
                )
            
            # If still no reviews, try loading from all_reviews.json
            if not reviews and metadata.get('all_reviews_path'):
                logging.info("Trying to load from all_reviews.json")
                extractor_class = load_module("5_extract_structured_reviews.py", class_name="ReviewExtractor")
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
                extractor_class = load_module("5_extract_structured_reviews.py", class_name="ReviewExtractor")
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
            
            if not reviews:
                logging.warning("No reviews found for this submission - skipping remaining steps")
                results['success'] = True
                return results
            
            # Step 6: Extract Claims from Reviews
            logging.info("Step 6: Extracting Claims from Reviews")
            claims_path = self.output_dir / "claims" / f"{results['submission_id']}_claims.json"
            reviews_with_claims = extract_claims_from_reviews(
                reviews,
                method=self.config.get("claim_extraction", "auto")
            )
            
            # Save claims
            with open(claims_path, 'w', encoding='utf-8') as f:
                json.dump(reviews_with_claims, f, indent=2, ensure_ascii=False)
            results['claims_path'] = str(claims_path)
            
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
            
            if not all_claims:
                logging.warning("No claims extracted from reviews")
                results['success'] = True
                return results
            
            # Step 7: Retrieve Evidence for Claims
            logging.info("Step 7: Retrieving Evidence for Claims")
            evidence_path = self.output_dir / "evidence" / f"{results['submission_id']}_evidence.json"
            
            # Load chunks
            retriever = EvidenceRetriever()
            chunks = retriever.load_chunks_from_jsonl(str(chunks_path))
            
            # Retrieve evidence for each claim
            claims_with_evidence = []
            for i, claim_data in enumerate(all_claims):
                evidence = retriever.retrieve_evidence(
                    claim_data['claim'],
                    chunks,
                    method=self.config.get("evidence_retrieval", "auto"),
                    top_k=self.config.get("top_k", 4)
                )
                
                claims_with_evidence.append({
                    'claim_id': i + 1,
                    'claim': claim_data['claim'],
                    'evidence': evidence,
                    'review_id': claim_data['review_id'],
                    'reviewer': claim_data['reviewer']
                })
            
            # Save evidence results
            with open(evidence_path, 'w', encoding='utf-8') as f:
                json.dump(claims_with_evidence, f, indent=2, ensure_ascii=False)
            results['evidence_path'] = str(evidence_path)
            
            # Step 8: Verify Claims
            logging.info("Step 8: Verifying Claims")
            verification_path = self.output_dir / "verification" / f"{results['submission_id']}_verification.json"
            report_path = self.output_dir / "verification" / f"{results['submission_id']}_report.md"
            
            verifier = ClaimVerifier()
            verification_results = verifier.verify_claims_batch(
                claims_with_evidence,
                model=self.config.get("verification_model", "qwen3:14b"),
                delay=1.0
            )
            
            # Save verification results
            verifier.save_verification_results(verification_results, str(verification_path))
            verifier.generate_verification_report(verification_results, str(report_path))
            
            results['verification_path'] = str(verification_path)
            results['report_path'] = str(report_path)
            results['success'] = True
            
            logging.info(f"Successfully processed submission {results['submission_id']}")
            
        except Exception as e:
            logging.error(f"Error processing URL {url}: {e}")
            results['error'] = str(e)
            results['success'] = False
        
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
            "claim_extraction": "auto",
            "evidence_retrieval": "auto",
            "verification_model": "qwen3:14b",
            "top_k": 4,
            "output_dir": "outputs"
        }
    
    # Create processor and run
    processor = OpenReviewProcessor(config)
    results = processor.process_openreview_url(args.url, args.username, args.password)
    
    # Print results
    if results['success']:
        print(f"\n✅ Successfully processed submission {results['submission_id']}")
        print(f"📄 PDF: {results['pdf_path']}")
        print(f"📝 Markdown: {results['markdown_path']}")
        print(f"🧹 Cleaned Markdown: {results['cleaned_markdown_path']}")
        print(f"📦 Chunks: {results['chunks_path']}")
        print(f"📋 Reviews: {results['reviews_path']}")
        print(f"🎯 Claims: {results['claims_path']}")
        print(f"🔍 Evidence: {results['evidence_path']}")
        print(f"✅ Verification: {results['verification_path']}")
        print(f"📊 Report: {results['report_path']}")
    else:
        print(f"\n❌ Failed to process URL: {results['error']}")

if __name__ == "__main__":
    main()
