"""
Structured Review Extraction Module
Extracts and structures review data from OpenReview submissions
"""

import json
import logging
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any

class ReviewExtractor:
    def __init__(self):
        """Initialize review extractor."""
        pass
    
    def extract_reviews_from_metadata(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract structured reviews from OpenReview metadata.
        
        Args:
            metadata (dict): OpenReview submission metadata
        
        Returns:
            list: List of structured reviews
        """
        reviews = []
        
        if 'reviews' not in metadata:
            logging.warning("No reviews found in metadata")
            return reviews
        
        for review in metadata['reviews']:
            structured_review = {
                'submission_id': metadata.get('submission_id'),
                'submission_title': metadata.get('title'),
                'review_id': review.get('review_id'),
                'reviewer': review.get('reviewer'),
                'invitation': review.get('invitation', ''),
                'rating': review.get('rating'),
                'confidence': review.get('confidence'),
                'review_text': review.get('review_text', ''),
                'summary': review.get('summary', ''),
                'strengths': review.get('strengths', ''),
                'weaknesses': review.get('weaknesses', ''),
                'questions': review.get('questions', ''),
                'limitations': review.get('limitations', ''),
                'soundness': review.get('soundness', ''),
                'presentation': review.get('presentation', ''),
                'contribution': review.get('contribution', ''),
                'flag_for_ethics_review': review.get('flag_for_ethics_review', ''),
                'details_of_ethics_concerns': review.get('details_of_ethics_concerns', ''),
                'full_review': self._combine_review_sections(review)
            }
            reviews.append(structured_review)
        
        return reviews
    
    def _combine_review_sections(self, review: Dict[str, Any]) -> str:
        """
        Combine different review sections into a single text.
        
        Args:
            review (dict): Review data
        
        Returns:
            str: Combined review text
        """
        sections = []
        
        # Add summary if available
        if review.get('summary'):
            sections.append(f"Summary: {review['summary']}")
        
        # Add main review text
        if review.get('review_text'):
            sections.append(review['review_text'])
        
        # Add strengths
        if review.get('strengths'):
            sections.append(f"Strengths: {review['strengths']}")
        
        # Add weaknesses
        if review.get('weaknesses'):
            sections.append(f"Weaknesses: {review['weaknesses']}")
        
        # Add questions
        if review.get('questions'):
            sections.append(f"Questions: {review['questions']}")
        
        # Add limitations
        if review.get('limitations'):
            sections.append(f"Limitations: {review['limitations']}")
        
        # Add soundness if available
        if review.get('soundness'):
            sections.append(f"Soundness: {review['soundness']}")
        
        # Add presentation if available
        if review.get('presentation'):
            sections.append(f"Presentation: {review['presentation']}")
        
        # Add contribution if available
        if review.get('contribution'):
            sections.append(f"Contribution: {review['contribution']}")
        
        # Add ethics concerns if flagged
        if review.get('flag_for_ethics_review'):
            sections.append(f"Ethics Review Flagged: {review['flag_for_ethics_review']}")
            if review.get('details_of_ethics_concerns'):
                sections.append(f"Ethics Concerns: {review['details_of_ethics_concerns']}")
        
        return '\n\n'.join(sections)
    
    def save_structured_reviews(self, reviews: List[Dict[str, Any]], output_path: str):
        """
        Save structured reviews to JSON file.
        
        Args:
            reviews (list): List of structured reviews
            output_path (str): Path to output JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved {len(reviews)} structured reviews to {output_path}")
    
    def process_metadata_file(self, metadata_path: str, output_path: str) -> List[Dict[str, Any]]:
        """
        Process metadata file and extract structured reviews.
        
        Args:
            metadata_path (str): Path to metadata JSON file
            output_path (str): Path to output structured reviews JSON file
        
        Returns:
            list: List of structured reviews
        """
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Extract reviews
        reviews = self.extract_reviews_from_metadata(metadata)
        
        # Save structured reviews
        self.save_structured_reviews(reviews, output_path)
        
        return reviews
    
    def filter_reviews_by_rating(self, reviews: List[Dict[str, Any]], min_rating: float = None, max_rating: float = None) -> List[Dict[str, Any]]:
        """
        Filter reviews by rating range.
        
        Args:
            reviews (list): List of reviews
            min_rating (float): Minimum rating (inclusive)
            max_rating (float): Maximum rating (inclusive)
        
        Returns:
            list: Filtered reviews
        """
        filtered_reviews = []
        
        for review in reviews:
            rating = review.get('rating')
            if rating is None:
                continue
            
            # Convert rating to float if it's a string
            try:
                rating = float(rating)
            except (ValueError, TypeError):
                continue
            
            # Apply filters
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue
            
            filtered_reviews.append(review)
        
        return filtered_reviews
    
    def get_review_statistics(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about reviews.
        
        Args:
            reviews (list): List of reviews
        
        Returns:
            dict: Review statistics
        """
        if not reviews:
            return {
                'total_reviews': 0,
                'average_rating': 0,
                'rating_distribution': {},
                'reviewers': set()
            }
        
        ratings = []
        reviewers = set()
        
        for review in reviews:
            rating = review.get('rating')
            if rating is not None:
                try:
                    ratings.append(float(rating))
                except (ValueError, TypeError):
                    pass
            
            reviewer = review.get('reviewer')
            if reviewer:
                reviewers.add(reviewer)
        
        # Calculate statistics
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Rating distribution
        rating_dist = {}
        for rating in ratings:
            rating_key = f"{int(rating)}-{int(rating)+1}"
            rating_dist[rating_key] = rating_dist.get(rating_key, 0) + 1
        
        return {
            'total_reviews': len(reviews),
            'average_rating': avg_rating,
            'rating_distribution': rating_dist,
            'reviewers': list(reviewers),
            'num_reviewers': len(reviewers)
        }
    
    def load_reviews_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load reviews from JSON file.
        
        Args:
            json_path (str): Path to JSON file containing reviews
        
        Returns:
            list: List of review dictionaries
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
            logging.info(f"Loaded {len(reviews)} reviews from {json_path}")
            return reviews
        except Exception as e:
            logging.error(f"Error loading reviews from {json_path}: {e}")
            return []
    
    def load_reviews_from_pickle(self, pickle_path: str) -> List[Dict[str, Any]]:
        """
        Load reviews from pickle file.
        
        Args:
            pickle_path (str): Path to pickle file containing reviews
        
        Returns:
            list: List of review dictionaries
        """
        try:
            with open(pickle_path, 'rb') as f:
                reviews = pickle.load(f)
            logging.info(f"Loaded {len(reviews)} reviews from {pickle_path}")
            return reviews
        except Exception as e:
            logging.error(f"Error loading reviews from {pickle_path}: {e}")
            return []
    
    def load_individual_review_files(self, reviews_dir: str, submission_id: str = None) -> List[Dict[str, Any]]:
        """
        Load individual review files from a directory.
        
        Args:
            reviews_dir (str): Directory containing individual review JSON files
            submission_id (str): Optional submission ID to filter files
        
        Returns:
            list: List of review dictionaries
        """
        reviews = []
        reviews_path = Path(reviews_dir)
        
        if not reviews_path.exists():
            logging.warning(f"Reviews directory does not exist: {reviews_dir}")
            return reviews
        
        # Pattern to match review files
        pattern = f"{submission_id}_review_*.json" if submission_id else "*_review_*.json"
        review_files = list(reviews_path.glob(pattern))
        
        logging.info(f"Found {len(review_files)} individual review files")
        
        for review_file in review_files:
            try:
                with open(review_file, 'r', encoding='utf-8') as f:
                    review = json.load(f)
                reviews.append(review)
            except Exception as e:
                logging.error(f"Error loading review file {review_file}: {e}")
        
        logging.info(f"Successfully loaded {len(reviews)} individual reviews")
        return reviews
    
    def process_individual_review(self, review_path: str) -> Dict[str, Any]:
        """
        Process a single review file independently.
        
        Args:
            review_path (str): Path to individual review JSON file
        
        Returns:
            dict: Processed review with additional metadata
        """
        try:
            with open(review_path, 'r', encoding='utf-8') as f:
                review = json.load(f)
            
            # Ensure the review has the full_review field
            if 'full_review' not in review or not review['full_review']:
                review['full_review'] = self._combine_review_sections(review)
            
            # Add file metadata
            review['source_file'] = review_path
            review['processed_timestamp'] = str(Path(review_path).stat().st_mtime)
            
            logging.info(f"Processed individual review: {review.get('review_id', 'unknown')}")
            return review
            
        except Exception as e:
            logging.error(f"Error processing review file {review_path}: {e}")
            return {}
    
    def process_all_individual_reviews(self, reviews_dir: str, submission_id: str = None) -> List[Dict[str, Any]]:
        """
        Process all individual review files in a directory.
        
        Args:
            reviews_dir (str): Directory containing individual review JSON files
            submission_id (str): Optional submission ID to filter files
        
        Returns:
            list: List of processed reviews
        """
        reviews = []
        reviews_path = Path(reviews_dir)
        
        if not reviews_path.exists():
            logging.warning(f"Reviews directory does not exist: {reviews_dir}")
            return reviews
        
        # Pattern to match review files
        pattern = f"{submission_id}_review_*.json" if submission_id else "*_review_*.json"
        review_files = list(reviews_path.glob(pattern))
        
        logging.info(f"Processing {len(review_files)} individual review files")
        
        for review_file in review_files:
            review = self.process_individual_review(str(review_file))
            if review:
                reviews.append(review)
        
        logging.info(f"Successfully processed {len(reviews)} individual reviews")
        return reviews

def extract_structured_reviews(metadata_path, output_path):
    """
    Main function to extract structured reviews from metadata file.
    
    Args:
        metadata_path (str): Path to metadata JSON file
        output_path (str): Path to output structured reviews JSON file
    
    Returns:
        list: List of structured reviews
    """
    extractor = ReviewExtractor()
    return extractor.process_metadata_file(metadata_path, output_path)

def extract_reviews_from_metadata(metadata):
    """
    Main function to extract structured reviews from metadata dict.
    
    Args:
        metadata (dict): OpenReview metadata
    
    Returns:
        list: List of structured reviews
    """
    extractor = ReviewExtractor()
    return extractor.extract_reviews_from_metadata(metadata)
