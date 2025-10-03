"""
Positional Chunking Module (V5 Column-Aware)
Generates chunks with precise positional data (bounding boxes) directly from a PDF.

This module uses Docling's structured export (export_to_dict) to access element-level
provenance (prov) including bounding boxes across pages/columns, and groups text into
semantically coherent, column-aware chunks suitable for frontend highlighting.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("Docling not available. Positional chunking requires Docling.")


class PositionalChunkerV5:
    def __init__(self, pdf_path: str | Path, output_dir: str | Path, column_split_x: float = 300.0):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.column_split_x = column_split_x
        self.docling_dict_output_path = self.output_dir / "docling_dict_output.json"
        self.docling_markdown_output_path = self.output_dir / "docling_markdown_output.md"

        self.docling_data = self._get_docling_data()
        self.elements = [
            el for el in self.docling_data.get('texts', [])
            if el.get('prov') and el.get('text', '').strip()
        ]

        # Sort elements by page, then by reading order (top->bottom, left->right)
        self.elements.sort(key=lambda x: (
            x['prov'][0]['page_no'],
            -x['prov'][0]['bbox']['t'],
            x['prov'][0]['bbox']['l']
        ))

    def _get_docling_data(self) -> Dict[str, Any]:
        """Parse PDF with Docling and return the structured dictionary output."""
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is required for positional chunking but is not installed.")

        if not self.docling_dict_output_path.exists():
            pipeline_options = PdfPipelineOptions()
            converter = DocumentConverter(format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            })
            result = converter.convert(str(self.pdf_path))

            # Persist markdown and dict for downstream usage/inspection
            self.docling_markdown_output_path.write_text(
                result.document.export_to_markdown(), encoding='utf-8'
            )
            doc_dict = result.document.export_to_dict()
            with open(self.docling_dict_output_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=2)
            return doc_dict

        with open(self.docling_dict_output_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_element_all_bbox_info(self, element: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all bounding boxes for an element across its prov entries."""
        prov = element.get('prov') or []
        bbox_infos: List[Dict[str, Any]] = []
        for prov_item in prov:
            bbox = prov_item.get('bbox')
            if not bbox:
                continue
            page_no = prov_item.get('page_no')
            center_x = (bbox['l'] + bbox['r']) / 2
            column = "left" if center_x < self.column_split_x else "right"
            bbox_infos.append({
                'page': page_no,
                'column': column,
                'bounding_box': {
                    'left': bbox['l'],
                    'top': bbox['t'],
                    'right': bbox['r'],
                    'bottom': bbox['b']
                },
                'char_span': prov_item.get('charspan')
            })
        return bbox_infos

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _create_chunk_with_all_positions(self, elements: List[Dict[str, Any]], chunk_id: str, chunk_type: str) -> Dict[str, Any]:
        text_parts: List[str] = []
        positions: List[Dict[str, Any]] = []
        pages = set()
        for el in elements:
            text_parts.append(el.get('text', ''))
            for pos in self._get_element_all_bbox_info(el):
                positions.append(pos)
                pages.add(pos['page'])

        full_text = "\n".join(text_parts)
        cols = {p['column'] for p in positions}
        return {
            'id': chunk_id,
            'type': chunk_type,
            'text': full_text,
            'tokens': self._estimate_tokens(full_text),
            'pages': sorted(pages),
            'positions': positions,
            'spans_columns': len(cols) > 1,
            'spans_pages': len(pages) > 1,
            'element_count': len(elements)
        }

    def chunk_column_aware_sections(self, max_tokens: int = 512) -> List[Dict[str, Any]]:
        """
        Create chunks by section headers while respecting column boundaries.
        Elements from different columns are not mixed in a single chunk.
        """
        chunks: List[Dict[str, Any]] = []
        current_section_name = "Introduction"
        current_section_elements: List[Dict[str, Any]] = []

        for element in self.elements:
            text = (element.get('text') or '').strip()
            if not text:
                continue

            bbox_infos = self._get_element_all_bbox_info(element)
            if not bbox_infos:
                continue

            label = element.get('label', 'text')
            is_section_header = label in ['section_header', 'page_header']

            if is_section_header:
                if current_section_elements:
                    column_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                    for el in current_section_elements:
                        bbox = self._get_element_all_bbox_info(el)
                        if bbox:
                            column_groups[bbox[0]['column']].append(el)
                    for col, col_elements in column_groups.items():
                        chunk_id = f"{current_section_name}_{col}_{len(chunks)}"
                        chunks.append(self._create_chunk_with_all_positions(col_elements, chunk_id, "column_aware_section"))

                current_section_elements = []
                current_section_name = text
                # Do not include the section header itself as a content element
                # Move to next element so headers never become standalone chunks
                continue

            current_section_elements.append(element)

            # Token-based split within current column
            primary_col = bbox_infos[0]['column']
            current_col_els = [
                el for el in current_section_elements
                if (self._get_element_all_bbox_info(el) and self._get_element_all_bbox_info(el)[0]['column'] == primary_col)
            ]
            total_tokens = sum(self._estimate_tokens(el.get('text', '')) for el in current_col_els)
            if total_tokens > max_tokens and len(current_col_els) > 1:
                to_chunk = current_col_els[:-1]
                chunk_id = f"{current_section_name}_{primary_col}_{len(chunks)}"
                chunks.append(self._create_chunk_with_all_positions(to_chunk, chunk_id, "column_aware_section"))
                current_section_elements = [el for el in current_section_elements if el not in to_chunk]

        if current_section_elements:
            column_groups = defaultdict(list)
            for el in current_section_elements:
                bbox = self._get_element_all_bbox_info(el)
                if bbox:
                    column_groups[bbox[0]['column']].append(el)
            for col, col_elements in column_groups.items():
                chunk_id = f"{current_section_name}_{col}_{len(chunks)}"
                chunks.append(self._create_chunk_with_all_positions(col_elements, chunk_id, "column_aware_section"))

        return chunks


def chunk_positional(pdf_path: str, output_dir: str, max_tokens: int = 512, column_split_x: float = 300.0) -> Dict[str, Any]:
    """
    Main function to generate positional chunks for a PDF.

    Args:
        pdf_path (str): Path to the input PDF
        output_dir (str): Output directory for artifacts
        max_tokens (int): Max tokens per chunk for grouping
        column_split_x (float): X coordinate dividing two columns

    Returns:
        dict: { 'chunks_path': str, 'num_chunks': int, 'docling_dict_path': str, 'docling_markdown_path': str }
    """
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    chunker = PositionalChunkerV5(pdf_path, output_dir_p, column_split_x=column_split_x)
    chunks = chunker.chunk_column_aware_sections(max_tokens=max_tokens)

    out_path = output_dir_p / "positional_chunks_v5.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logging.info(f"Generated {len(chunks)} positional chunks → {out_path}")

    return {
        'chunks_path': str(out_path),
        'num_chunks': len(chunks),
        'docling_dict_path': str(chunker.docling_dict_output_path),
        'docling_markdown_path': str(chunker.docling_markdown_output_path)
    }


