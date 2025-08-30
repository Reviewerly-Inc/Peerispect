"""
OpenReview Crawler Module
Extracts paper PDF and reviews from a single OpenReview URL
"""

import openreview
from openreview import api
import os
import json
import logging
import time
import requests
from urllib.parse import urlparse
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class OpenReviewCrawler:
    def __init__(self, username=None, password=None):
        """Initialize OpenReview client."""
        self.client = api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=username,
            password=password
        )
    
    def extract_submission_id_from_url(self, url):
        """Extract submission ID from OpenReview URL."""
        # Handle different URL formats
        if 'id=' in url:
            return url.split('id=')[1].split('&')[0]
        elif '/forum/' in url:
            return url.split('/forum/')[1].split('/')[0]
        else:
            raise ValueError("Could not extract submission ID from URL")
    
    def get_submission_data(self, submission_id, retries=5):
        """Fetch submission data with retry mechanism using details=all."""
        for attempt in range(retries):
            try:
                logging.info(f"Fetching submission: {submission_id} (Attempt {attempt + 1}/{retries})")
                # Use get_all_notes with details='all' to get complete data including reviews
                submissions = self.client.get_all_notes(
                    id=submission_id,
                    details='all'
                )
                if submissions:
                    return submissions[0]  # Return the first (and should be only) submission
                else:
                    logging.warning(f"No submission found with ID: {submission_id}")
                    return None
            except openreview.OpenReviewException as e:
                if "RateLimitError" in str(e):
                    retry_after = 30
                    logging.warning(f"Rate limit reached. Retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                else:
                    logging.error(f"OpenReviewException: {e}")
                    raise
            except Exception as e:
                logging.error(f"General error while fetching submission {submission_id}: {e}")
                raise
        logging.error(f"Failed to fetch submission {submission_id} after {retries} attempts.")
        return None
    
    def get_reviews_for_submission(self, submission_id, retries=5):
        """Fetch all reviews for a submission using the OpenReview API."""
        for attempt in range(retries):
            try:
                logging.info(f"Fetching reviews for submission: {submission_id} (Attempt {attempt + 1}/{retries})")
                
                # Get all notes that are replies to this submission
                replies = self.client.get_all_notes(forum=submission_id)
                
                reviews = []
                for reply in replies:
                    # Check if this is a review (has invitation containing 'review' and is not the main submission)
                    if (hasattr(reply, 'invitation') and 
                        reply.invitation and 
                        'review' in reply.invitation.lower() and 
                        reply.id != submission_id):
                        
                        def get_value(field):
                            """Extract value from OpenReview content field."""
                            if not hasattr(reply, 'content') or not reply.content:
                                return ''
                            v = reply.content.get(field)
                            if isinstance(v, dict):
                                return v.get('value', v)
                            return v or ''
                        
                        review_info = {
                            'review_id': reply.id,
                            'reviewer': reply.signatures[0] if hasattr(reply, 'signatures') and reply.signatures else 'Anonymous',
                            'invitation': reply.invitation if hasattr(reply, 'invitation') else '',
                            'rating': get_value('rating'),
                            'confidence': get_value('confidence'),
                            'review_text': get_value('review'),
                            'summary': get_value('summary'),
                            'strengths': get_value('strengths'),
                            'weaknesses': get_value('weaknesses'),
                            'questions': get_value('questions'),
                            'limitations': get_value('limitations'),
                            'soundness': get_value('soundness'),
                            'presentation': get_value('presentation'),
                            'contribution': get_value('contribution'),
                            'flag_for_ethics_review': get_value('flag_for_ethics_review'),
                            'details_of_ethics_concerns': get_value('details_of_ethics_concerns')
                        }
                        reviews.append(review_info)
                
                logging.info(f"Found {len(reviews)} reviews for submission {submission_id}")
                return reviews
                
            except openreview.OpenReviewException as e:
                if "RateLimitError" in str(e):
                    retry_after = 30
                    logging.warning(f"Rate limit reached. Retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                else:
                    logging.error(f"OpenReviewException while fetching reviews: {e}")
                    if attempt == retries - 1:
                        raise
            except Exception as e:
                logging.error(f"General error while fetching reviews for {submission_id}: {e}")
                if attempt == retries - 1:
                    raise
        
        logging.error(f"Failed to fetch reviews for {submission_id} after {retries} attempts.")
        return []
    
    def download_pdf(self, pdf_url, pdf_path, retries=3):
        """Download a PDF file from URL with retry mechanism."""
        for attempt in range(retries):
            try:
                logging.info(f"Downloading PDF: {pdf_url} (Attempt {attempt + 1}/{retries})")
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Successfully downloaded PDF to: {pdf_path}")
                return True
            except requests.exceptions.RequestException as e:
                logging.warning(f"Failed to download PDF (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    logging.error(f"Failed to download PDF after {retries} attempts: {pdf_url}")
                    return False
        return False
    
    def extract_pdf_info(self, submission):
        """Extract PDF information from a submission."""
        def get_value(field):
            v = submission.content.get(field)
            if isinstance(v, dict):
                return v.get('value', v)
            return v

        pdf_info = {
            'submission_id': submission.id,
            'title': get_value('title'),
            'authors': get_value('authors'),
            'abstract': get_value('abstract'),
            'pdf_url': None,
            'pdf_filename': None,
            'reviews': []
        }

        # Extract PDF URL from submission
        if 'pdf' in submission.content:
            pdf_url = get_value('pdf')
            if pdf_url:
                if pdf_url.startswith('/'):
                    pdf_url = f"https://openreview.net/pdf?id={submission.id}"
                elif not pdf_url.startswith('http'):
                    pdf_url = f"https://openreview.net/pdf?id={submission.id}"
                
                pdf_info['pdf_url'] = pdf_url
                filename = f"{submission.id}.pdf"
                pdf_info['pdf_filename'] = filename

        # Extract reviews using the proven approach from working code
        # Look for replies in submission.details['replies'] (available with details='all')
        if hasattr(submission, 'details') and submission.details.get('replies'):
            replies = submission.details['replies']
            logging.info(f"Found {len(replies)} replies for submission {submission.id}")
            
            # Filter for official reviews using the proven logic
            for reply in replies:
                # Check if this is an official review
                invitations = reply.get('invitations', [])
                signatures = reply.get('signatures', [])
                
                # Check if any invitation contains 'Review' and signature contains 'Reviewer_'
                is_review = any('Review' in inv for inv in invitations)
                is_from_reviewer = any('Reviewer_' in sig for sig in signatures)
                
                if is_review and is_from_reviewer:
                    logging.info(f"Found official review: {reply.get('id')} with invitations: {invitations}")
                    
                    content = reply.get('content', {})
                    
                    def reply_get_value(field):
                        v = content.get(field, '')
                        if isinstance(v, dict):
                            return v.get('value', v)
                        return v
                    
                    # Extract reviewer name from signature
                    reviewer = 'Anonymous'
                    if signatures:
                        reviewer = signatures[0].split('/')[-1] if signatures[0] else 'Anonymous'
                    
                    # Combine all review content as in the working code
                    review_text_parts = []
                    for field_name, field_value in content.items():
                        if isinstance(field_value, dict) and 'value' in field_value:
                            value = field_value['value']
                            if isinstance(value, str) and value.strip():
                                review_text_parts.append(value)
                    
                    combined_review_text = '\n'.join(review_text_parts).strip()
                    
                    review_info = {
                        'review_id': reply.get('id'),
                        'reviewer': reviewer,
                        'invitation': invitations[0] if invitations else '',
                        'invitations': invitations,
                        'signatures': signatures,
                        'rating': reply_get_value('rating'),
                        'confidence': reply_get_value('confidence'),
                        'review_text': reply_get_value('review'),
                        'summary': reply_get_value('summary'),
                        'strengths': reply_get_value('strengths'),
                        'weaknesses': reply_get_value('weaknesses'),
                        'questions': reply_get_value('questions'),
                        'limitations': reply_get_value('limitations'),
                        'soundness': reply_get_value('soundness'),
                        'presentation': reply_get_value('presentation'),
                        'contribution': reply_get_value('contribution'),
                        'flag_for_ethics_review': reply_get_value('flag_for_ethics_review'),
                        'details_of_ethics_concerns': reply_get_value('details_of_ethics_concerns'),
                        'combined_review_text': combined_review_text  # Add the combined text as in working code
                    }
                    pdf_info['reviews'].append(review_info)
                    logging.info(f"Extracted review {review_info['review_id']} from {reviewer} with rating: {review_info['rating']}")
                else:
                    # Log non-review replies for debugging
                    logging.debug(f"Non-review reply {reply.get('id')} with invitations: {invitations}")
            
            logging.info(f"Successfully extracted {len(pdf_info['reviews'])} reviews for submission {submission.id}")
        else:
            logging.warning(f"No replies found in submission.details for {submission.id}")
            
            # Fallback: try the API method as backup
            try:
                logging.info("Trying API method as fallback...")
                reviews = self.get_reviews_for_submission(submission.id)
                pdf_info['reviews'] = reviews
                logging.info(f"API fallback method found {len(reviews)} reviews")
            except Exception as e:
                logging.warning(f"API fallback method also failed: {e}")
                pdf_info['reviews'] = []

        return pdf_info
    
    def process_single_url(self, url, output_dir="outputs"):
        """Process a single OpenReview URL to extract paper and reviews."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        reviews_dir = os.path.join(output_dir, "reviews")
        pdfs_dir = os.path.join(output_dir, "pdfs")
        os.makedirs(reviews_dir, exist_ok=True)
        os.makedirs(pdfs_dir, exist_ok=True)
        
        # Extract submission ID
        submission_id = self.extract_submission_id_from_url(url)
        logging.info(f"Processing submission ID: {submission_id}")
        
        # Get submission data
        submission = self.get_submission_data(submission_id)
        if not submission:
            raise ValueError(f"Could not fetch submission {submission_id}")
        
        # Extract PDF info
        pdf_info = self.extract_pdf_info(submission)
        
        # Download PDF if available
        if pdf_info['pdf_url']:
            pdf_path = os.path.join(pdfs_dir, pdf_info['pdf_filename'])
            if self.download_pdf(pdf_info['pdf_url'], pdf_path):
                pdf_info['pdf_local_path'] = pdf_path
            else:
                pdf_info['pdf_local_path'] = None
        else:
            pdf_info['pdf_local_path'] = None
            logging.warning(f"No PDF URL found for submission {submission_id}")
        
        # Save individual reviews as separate files
        if pdf_info['reviews']:
            # Save all reviews in one JSON file
            all_reviews_path = os.path.join(reviews_dir, f"{submission_id}_all_reviews.json")
            with open(all_reviews_path, 'w', encoding='utf-8') as f:
                json.dump(pdf_info['reviews'], f, indent=2, ensure_ascii=False)
            logging.info(f"Saved {len(pdf_info['reviews'])} reviews to {all_reviews_path}")
            
            # Save all reviews as pickle for backup
            import pickle
            all_reviews_pickle_path = os.path.join(reviews_dir, f"{submission_id}_all_reviews.pkl")
            with open(all_reviews_pickle_path, 'wb') as f:
                pickle.dump(pdf_info['reviews'], f)
            logging.info(f"Saved reviews pickle backup to {all_reviews_pickle_path}")
            
            # Save each review as an individual JSON file for independent processing
            for i, review in enumerate(pdf_info['reviews']):
                review_filename = f"{submission_id}_review_{i+1}_{review.get('review_id', 'unknown')}.json"
                # Clean filename to remove invalid characters
                review_filename = re.sub(r'[<>:"/\\|?*]', '_', review_filename)
                review_path = os.path.join(reviews_dir, review_filename)
                
                # Add metadata to individual review
                individual_review = {
                    'submission_id': submission_id,
                    'submission_title': pdf_info.get('title'),
                    'review_index': i + 1,
                    **review
                }
                
                with open(review_path, 'w', encoding='utf-8') as f:
                    json.dump(individual_review, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved {len(pdf_info['reviews'])} individual review files")
            
            # Update pdf_info with review file paths
            pdf_info['all_reviews_path'] = all_reviews_path
            pdf_info['all_reviews_pickle_path'] = all_reviews_pickle_path
            pdf_info['individual_review_files'] = [
                os.path.join(reviews_dir, f"{submission_id}_review_{i+1}_{review.get('review_id', 'unknown')}.json".replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_'))
                for i, review in enumerate(pdf_info['reviews'])
            ]
        else:
            logging.warning(f"No reviews found for submission {submission_id}")
            pdf_info['all_reviews_path'] = None
            pdf_info['all_reviews_pickle_path'] = None
            pdf_info['individual_review_files'] = []
        
        # Save metadata in pdfs directory
        metadata_path = os.path.join(pdfs_dir, f"{submission_id}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_info, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved metadata to {metadata_path}")
        return pdf_info

def crawl_openreview_url(url, output_dir="outputs", username=None, password=None):
    """
    Main function to crawl a single OpenReview URL.
    
    Args:
        url (str): OpenReview URL
        output_dir (str): Directory to save outputs
        username (str): OpenReview username (optional)
        password (str): OpenReview password (optional)
    
    Returns:
        dict: Extracted paper and review information
    """
    crawler = OpenReviewCrawler(username=username, password=password)
    return crawler.process_single_url(url, output_dir)
