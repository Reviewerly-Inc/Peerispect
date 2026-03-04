"""
Markdown Cleaning Module
Cleans and formats parsed markdown content
"""

import re
import os
import logging

class MarkdownCleaner:
    def __init__(self):
        """Initialize markdown cleaner."""
        pass
    
    def clean_text_remove_long_repeats(self, text: str, char_thresh: int = 10, punct_seq_thresh: int = 10) -> str:
        """
        Remove long repeated characters and punctuation in the given text.
        Sequences longer than the threshold are truncated to the threshold length.
        """
        # Replace characters repeated more than char_thresh times
        text = re.sub(r"(.)\1{" + str(char_thresh) + r",}", lambda m: m.group(1) * char_thresh, text)
        
        # Replace punctuation characters repeated more than punct_seq_thresh times
        text = re.sub(r"([^\w\s])\1{" + str(punct_seq_thresh) + r",}", lambda m: m.group(1) * punct_seq_thresh, text)
        
        return text
    
    def remove_neurips_checklist(self, text: str) -> str:
        """Remove NeurIPS Paper Checklist section."""
        keyword = "NeurIPS Paper Checklist"
        index = text.find(keyword)
        if index != -1:
            return text[:index].rstrip()
        return text
    
    def remove_iclr_checklist(self, text: str) -> str:
        """Remove ICLR Paper Checklist section."""
        keyword = "ICLR Paper Checklist"
        index = text.find(keyword)
        if index != -1:
            return text[:index].rstrip()
        return text
    
    def remove_common_checklists(self, text: str) -> str:
        """Remove common conference checklists."""
        checklists = [
            "NeurIPS Paper Checklist",
            "ICLR Paper Checklist", 
            "Paper Checklist",
            "Submission Checklist"
        ]
        
        for checklist in checklists:
            index = text.find(checklist)
            if index != -1:
                return text[:index].rstrip()
        
        return text
    
    def clean_markdown_formatting(self, text: str) -> str:
        """Clean markdown formatting issues."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix broken headers
        text = re.sub(r'^#+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove empty lines at start and end
        text = text.strip()
        
        # Fix broken code blocks
        text = re.sub(r'```\s*\n\s*```', '', text)
        
        return text
    
    def remove_figure_placeholders(self, text: str) -> str:
        """Remove figure placeholders and image references."""
        # Remove markdown image syntax
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        
        # Remove figure placeholders
        text = re.sub(r'\[Figure \d+\]', '', text)
        text = re.sub(r'Figure \d+', '', text)
        
        # Remove image placeholders
        text = re.sub(r'\[Image \d+\]', '', text)
        text = re.sub(r'Image \d+', '', text)
        
        return text
    
    def clean_special_characters(self, text: str) -> str:
        """Clean special characters and encoding issues."""
        # Replace common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('â€"', '—')
        text = text.replace('â€¦', '...')
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    def clean_markdown_file(self, input_path: str, output_path: str = None, 
                          char_thresh: int = 10, punct_seq_thresh: int = 10,
                          remove_checklists: bool = True, remove_figures: bool = True) -> str:
        """
        Clean a markdown file and save the result.
        
        Args:
            input_path (str): Path to input markdown file
            output_path (str): Path to output cleaned markdown file (optional)
            char_thresh (int): Threshold for repeated characters
            punct_seq_thresh (int): Threshold for repeated punctuation
            remove_checklists (bool): Whether to remove conference checklists
            remove_figures (bool): Whether to remove figure placeholders
        
        Returns:
            str: Path to cleaned markdown file
        """
        if output_path is None:
            output_path = input_path
        
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply cleaning steps
        logging.info("Cleaning markdown content...")
        
        # Remove long repeats
        content = self.clean_text_remove_long_repeats(content, char_thresh, punct_seq_thresh)
        
        # Remove checklists if requested
        if remove_checklists:
            content = self.remove_common_checklists(content)
        
        # Remove figures if requested
        if remove_figures:
            content = self.remove_figure_placeholders(content)
        
        # Clean formatting
        content = self.clean_markdown_formatting(content)
        
        # Clean special characters
        content = self.clean_special_characters(content)
        
        # Write cleaned content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logging.info(f"Cleaned markdown saved to: {output_path}")
        return output_path
    
    def clean_markdown_text(self, text: str, char_thresh: int = 10, punct_seq_thresh: int = 10,
                          remove_checklists: bool = True, remove_figures: bool = True) -> str:
        """
        Clean markdown text directly.
        
        Args:
            text (str): Input markdown text
            char_thresh (int): Threshold for repeated characters
            punct_seq_thresh (int): Threshold for repeated punctuation
            remove_checklists (bool): Whether to remove conference checklists
            remove_figures (bool): Whether to remove figure placeholders
        
        Returns:
            str: Cleaned markdown text
        """
        # Apply cleaning steps
        text = self.clean_text_remove_long_repeats(text, char_thresh, punct_seq_thresh)
        
        if remove_checklists:
            text = self.remove_common_checklists(text)
        
        if remove_figures:
            text = self.remove_figure_placeholders(text)
        
        text = self.clean_markdown_formatting(text)
        text = self.clean_special_characters(text)
        
        return text

def clean_markdown(input_path, output_path=None, **kwargs):
    """
    Main function to clean markdown file.
    
    Args:
        input_path (str): Path to input markdown file
        output_path (str): Path to output cleaned markdown file
        **kwargs: Additional cleaning options
    
    Returns:
        str: Path to cleaned markdown file
    """
    cleaner = MarkdownCleaner()
    return cleaner.clean_markdown_file(input_path, output_path, **kwargs)

def clean_markdown_text(text, **kwargs):
    """
    Main function to clean markdown text.
    
    Args:
        text (str): Input markdown text
        **kwargs: Additional cleaning options
    
    Returns:
        str: Cleaned markdown text
    """
    cleaner = MarkdownCleaner()
    return cleaner.clean_markdown_text(text, **kwargs)
