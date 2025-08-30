"""
PDF Parser Module
Parses PDF files to markdown using multiple methods
"""

import os
import sys
import time
import json
import logging
from typing import List
from pathlib import Path

# Try to import advanced PDF parsing libraries
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("Docling not available. Using fallback PDF parsing.")

try:
    import torch
    from transformers import AutoProcessor, VisionEncoderDecoderModel
    from pdf2image import convert_from_path
    NOUGAT_AVAILABLE = True
except ImportError:
    NOUGAT_AVAILABLE = False
    logging.warning("Nougat dependencies not available.")

# Fallback PDF parsing
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    try:
        import pypdf
        PYPDF2_AVAILABLE = True
    except ImportError:
        PYPDF2_AVAILABLE = False
        logging.warning("PyPDF2/pypdf not available for fallback parsing.")

class PDFParser:
    def __init__(self):
        """Initialize PDF parser with available methods."""
        self.available_methods = []
        
        if DOCLING_AVAILABLE:
            self.available_methods.append("docling_standard")
            self.available_methods.append("docling_vllm")
        
        if NOUGAT_AVAILABLE:
            self.available_methods.append("nougat")
        
        if PYPDF2_AVAILABLE:
            self.available_methods.append("pypdf2_fallback")
        
        logging.info(f"Available PDF parsing methods: {self.available_methods}")
    
    def parse_with_docling_standard(self, file_path, output_path, code_enrichment=False, formula_enrichment=False):
        """Parse PDF using Docling standard pipeline."""
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling not available")
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_code_enrichment = code_enrichment
        pipeline_options.do_formula_enrichment = formula_enrichment

        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        })
        
        result = converter.convert(file_path)
        
        with open(output_path, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write(result.document.export_to_markdown())
        
        return output_path
    
    def parse_with_nougat(self, file_path, output_path, model="0.1.0-small"):
        """Parse PDF using Nougat."""
        if not NOUGAT_AVAILABLE:
            raise ImportError("Nougat dependencies not available")
        
        def _resolve_nougat_tag(model_tag: str) -> str:
            mapping = {
                "0.1.0-small": "facebook/nougat-small",
                "0.1.0-base": "facebook/nougat-base",
                "nougat-small": "facebook/nougat-small",
                "nougat-base": "facebook/nougat-base",
            }
            if model_tag not in mapping:
                raise ValueError(f"Unsupported model '{model_tag}'. Choose one of: {list(mapping)}")
            return mapping[model_tag]
        
        def _load_nougat(model_tag: str):
            name = _resolve_nougat_tag(model_tag)
            processor = AutoProcessor.from_pretrained(name)
            model_obj = VisionEncoderDecoderModel.from_pretrained(name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_obj.to(device)
            model_obj.eval()
            return processor, model_obj, device
        
        def _markdown_from_images(images: List, processor, model, device, batch_size: int = 1) -> str:
            markdown_parts: List[str] = []
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i : i + batch_size]
                pix = processor(images=batch_imgs, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        pix,
                        min_length=1,
                        max_new_tokens=4096,
                        bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    )
                batch_text = processor.batch_decode(outputs, skip_special_tokens=True)
                batch_text = [processor.post_process_generation(t, fix_markdown=True) for t in batch_text]
                markdown_parts.extend(batch_text)
            
            markdown_clean = "\n".join([line for line in markdown_parts if not line.strip().startswith("![")])
            return markdown_clean
        
        file_path = Path(file_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        processor, model_obj, device = _load_nougat(model)
        images = convert_from_path(str(file_path), dpi=300)
        markdown = _markdown_from_images(images, processor, model_obj, device)
        
        output_path.write_text(markdown, encoding="utf-8")
        return str(output_path)
    
    def parse_with_pypdf2_fallback(self, file_path, output_path):
        """Parse PDF using PyPDF2/pypdf as fallback."""
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2/pypdf not available")
        
        try:
            import PyPDF2
            reader_class = PyPDF2.PdfReader
        except ImportError:
            import pypdf
            reader_class = pypdf.PdfReader
        
        text_content = []
        
        with open(file_path, 'rb') as file:
            reader = reader_class(file)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content.append(f"## Page {page_num + 1}\n\n{text}\n")
        
        markdown_content = "\n".join(text_content)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return output_path
    
    def parse_pdf(self, file_path, output_path, method="auto", **kwargs):
        """
        Parse PDF file to markdown using specified method.
        
        Args:
            file_path (str): Path to PDF file
            output_path (str): Path to output markdown file
            method (str): Parsing method ("auto", "docling_standard", "nougat", "pypdf2_fallback")
            **kwargs: Additional arguments for specific methods
        
        Returns:
            str: Path to output markdown file
        """
        if method == "auto":
            # Try methods in order of preference
            for preferred_method in ["docling_standard", "nougat", "pypdf2_fallback"]:
                if preferred_method in self.available_methods:
                    method = preferred_method
                    break
            else:
                raise ValueError("No PDF parsing methods available")
        
        if method not in self.available_methods:
            raise ValueError(f"Method '{method}' not available. Available methods: {self.available_methods}")
        
        logging.info(f"Parsing PDF with method: {method}")
        start_time = time.time()
        
        try:
            if method == "docling_standard":
                result = self.parse_with_docling_standard(
                    file_path, 
                    output_path,
                    code_enrichment=kwargs.get('code_enrichment', False),
                    formula_enrichment=kwargs.get('formula_enrichment', False)
                )
            elif method == "nougat":
                result = self.parse_with_nougat(
                    file_path,
                    output_path,
                    model=kwargs.get('model', "0.1.0-small")
                )
            elif method == "pypdf2_fallback":
                result = self.parse_with_pypdf2_fallback(file_path, output_path)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            elapsed_time = time.time() - start_time
            logging.info(f"PDF parsing completed in {elapsed_time:.2f} seconds")
            return result
            
        except Exception as e:
            logging.error(f"Error parsing PDF with method {method}: {e}")
            raise

def parse_pdf_to_markdown(pdf_path, output_path, method="auto", **kwargs):
    """
    Main function to parse PDF to markdown.
    
    Args:
        pdf_path (str): Path to PDF file
        output_path (str): Path to output markdown file
        method (str): Parsing method
        **kwargs: Additional arguments for specific methods
    
    Returns:
        str: Path to output markdown file
    """
    parser = PDFParser()
    return parser.parse_pdf(pdf_path, output_path, method, **kwargs)
