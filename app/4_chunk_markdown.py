"""
Markdown Chunking Module
Chunks markdown files into manageable pieces for retrieval
"""

import json
import re
from pathlib import Path
from typing import List
import logging

class MarkdownChunker:
    def __init__(self, encoding_name="cl100k_base"):
        """
        Initialize markdown chunker.
        
        Args:
            encoding_name (str): Tokenizer encoding name (not used in fallback mode)
        """
        self.encoding = None
        logging.info("Using fallback tokenization (character-based)")
    
    def num_tokens(self, text: str) -> int:
        """Count tokens in text using character-based approximation."""
        # Fallback: approximate tokens as characters / 4
        return len(text) // 4
    
    def chunk_document(self, text: str, max_tokens: int) -> List[str]:
        """
        Split text (markdown) into chunks of ≤ max_tokens, keeping paragraphs intact.
        If a single paragraph exceeds max_tokens, it will be split on sentence boundaries.
        """
        # Split on blank lines → paragraphs
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for p in paras:
            p_tokens = self.num_tokens(p)
            
            # Case: paragraph itself too big → split by sentences
            if p_tokens > max_tokens:
                # Flush existing
                if current:
                    chunks.append("\n\n".join(current))
                    current, current_tokens = [], 0

                # Break p into sentences
                sents = re.split(r'(?<=[\.\?\!])\s+', p)
                buf: List[str] = []
                buf_tokens = 0
                
                for s in sents:
                    s_tokens = self.num_tokens(s)
                    # If adding fits, do it
                    if buf_tokens + s_tokens <= max_tokens:
                        buf.append(s)
                        buf_tokens += s_tokens
                    else:
                        # Flush buffer
                        if buf:
                            chunks.append(" ".join(buf))
                        buf = [s]
                        buf_tokens = s_tokens
                
                if buf:
                    chunks.append(" ".join(buf))
                continue

            # Normal case: paragraph fits
            if current_tokens + p_tokens <= max_tokens:
                current.append(p)
                current_tokens += p_tokens
            else:
                # Flush current chunk
                chunks.append("\n\n".join(current))
                current = [p]
                current_tokens = p_tokens

        # Final flush
        if current:
            chunks.append("\n\n".join(current))

        return chunks
    
    def chunk_markdown_file(self, input_path: str, output_path: str, max_tokens: int = 512):
        """
        Chunk a markdown file and save as JSONL.
        
        Args:
            input_path (str): Path to input markdown file
            output_path (str): Path to output JSONL file
            max_tokens (int): Maximum tokens per chunk
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read markdown file
        text = input_path.read_text(encoding="utf-8")
        chunks = self.chunk_document(text, max_tokens)

        # Create record
        record = {
            "file_id": input_path.stem,
            "chunks": [
                {"idx": i + 1, "text": chunks[i]}
                for i in range(len(chunks))
            ]
        }

        # Write JSONL
        with output_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logging.info(f"Chunked {len(chunks)} chunks from {input_path} to {output_path}")
        return record
    
    def chunk_markdown_text(self, text: str, file_id: str = "document", max_tokens: int = 512):
        """
        Chunk markdown text directly.
        
        Args:
            text (str): Input markdown text
            file_id (str): Identifier for the document
            max_tokens (int): Maximum tokens per chunk
        
        Returns:
            dict: Chunked document record
        """
        chunks = self.chunk_document(text, max_tokens)
        
        record = {
            "file_id": file_id,
            "chunks": [
                {"idx": i + 1, "text": chunks[i]}
                for i in range(len(chunks))
            ]
        }
        
        return record
    
    def chunk_directory(self, input_dir: str, output_dir: str, max_tokens: int = 512):
        """
        Chunk all markdown files in a directory.
        
        Args:
            input_dir (str): Input directory containing .md files
            output_dir (str): Output directory for JSONL files
            max_tokens (int): Maximum tokens per chunk
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        md_files = list(input_path.glob("*.md"))
        logging.info(f"Found {len(md_files)} markdown files to chunk")
        
        for md_file in md_files:
            try:
                output_file = output_path / f"{md_file.stem}_chunks.jsonl"
                self.chunk_markdown_file(md_file, output_file, max_tokens)
            except Exception as e:
                logging.error(f"Error chunking {md_file}: {e}")

def chunk_markdown(input_path, output_path=None, max_tokens=512):
    """
    Main function to chunk markdown file.
    
    Args:
        input_path (str): Path to input markdown file
        output_path (str): Path to output JSONL file
        max_tokens (int): Maximum tokens per chunk
    
    Returns:
        dict: Chunked document record
    """
    chunker = MarkdownChunker()
    return chunker.chunk_markdown_file(input_path, output_path, max_tokens)

def chunk_markdown_text(text, file_id="document", max_tokens=512):
    """
    Main function to chunk markdown text.
    
    Args:
        text (str): Input markdown text
        file_id (str): Document identifier
        max_tokens (int): Maximum tokens per chunk
    
    Returns:
        dict: Chunked document record
    """
    chunker = MarkdownChunker()
    return chunker.chunk_markdown_text(text, file_id, max_tokens)

def chunk_markdown_directory(input_dir, output_dir, max_tokens=512):
    """
    Main function to chunk all markdown files in a directory.
    
    Args:
        input_dir (str): Input directory
        output_dir (str): Output directory
        max_tokens (int): Maximum tokens per chunk
    """
    chunker = MarkdownChunker()
    chunker.chunk_directory(input_dir, output_dir, max_tokens)
