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
        
        # Add tables to elements
        self.tables = [
            table for table in self.docling_data.get('tables', [])
            if table.get('prov') and self._extract_table_text(table).strip()
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

    def _extract_table_text(self, table: Dict[str, Any]) -> str:
        """Extract text content from table structure."""
        text_parts = []
        
        # Get table data
        data = table.get('data', {})
        if not data:
            return ""
        
        # Extract text from table cells
        table_cells = data.get('table_cells', [])
        if not table_cells:
            return ""
        
        # Group cells by row
        rows = {}
        for cell in table_cells:
            row_idx = cell.get('start_row_offset_idx', 0)
            if row_idx not in rows:
                rows[row_idx] = []
            rows[row_idx].append(cell)
        
        # Sort cells by column within each row
        for row_idx in rows:
            rows[row_idx].sort(key=lambda x: x.get('start_col_offset_idx', 0))
        
        # Build table text
        for row_idx in sorted(rows.keys()):
            row_cells = rows[row_idx]
            row_text = []
            for cell in row_cells:
                cell_text = cell.get('text', '').strip()
                if cell_text:
                    row_text.append(cell_text)
            if row_text:
                text_parts.append(' | '.join(row_text))
        
        return '\n'.join(text_parts)

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
        Includes table processing.
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

        # Process tables as separate chunks
        table_chunk_id = 1000  # Start table chunks with higher ID
        for table in self.tables:
            table_text = self._extract_table_text(table)
            if not table_text.strip():
                continue
            
            bbox_infos = self._get_element_all_bbox_info(table)
            if not bbox_infos:
                continue
            
            # Create table chunk
            chunk = {
                'id': f'table_{table_chunk_id}',
                'type': 'table',
                'text': table_text,
                'tokens': self._estimate_tokens(table_text),
                'pages': sorted(list(set([bbox['page'] for bbox in bbox_infos]))),
                'positions': bbox_infos,
                'spans_columns': len(set([bbox['column'] for bbox in bbox_infos])) > 1,
                'spans_pages': len(set([bbox['page'] for bbox in bbox_infos])) > 1,
                'element_count': 1
            }
            
            chunks.append(chunk)
            table_chunk_id += 1

        # Post-process: merge chunks that are too small (< 100 tokens)
        chunks = self._merge_small_chunks(chunks, min_tokens=100)
        
        return chunks

    def _merge_small_chunks(self, chunks: List[Dict[str, Any]], min_tokens: int = 100) -> List[Dict[str, Any]]:
        """
        Merge chunks that are smaller than min_tokens with their neighbors.
        Prefers merging with the shorter neighbor to balance chunk sizes.
        """
        if not chunks:
            return chunks
        
        result = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # If current chunk is large enough, keep it as is
            if current_chunk.get('tokens', 0) >= min_tokens:
                result.append(current_chunk)
                i += 1
                continue
            
            # Current chunk is too small, need to merge
            # Find the best neighbor to merge with
            prev_chunk = result[-1] if result else None
            next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
            
            # Decide which neighbor to merge with
            if prev_chunk is None and next_chunk is None:
                # No neighbors, keep as is (shouldn't happen)
                result.append(current_chunk)
                i += 1
            elif prev_chunk is None:
                # Only next chunk available, merge forward
                merged_chunk = self._merge_two_chunks(current_chunk, next_chunk)
                result.append(merged_chunk)
                i += 2  # Skip next chunk since we merged it
            elif next_chunk is None:
                # Only previous chunk available, merge backward
                merged_chunk = self._merge_two_chunks(prev_chunk, current_chunk)
                result[-1] = merged_chunk  # Replace the last chunk
                i += 1
            else:
                # Both neighbors available, choose the shorter one
                if prev_chunk.get('tokens', 0) <= next_chunk.get('tokens', 0):
                    # Merge with previous chunk
                    merged_chunk = self._merge_two_chunks(prev_chunk, current_chunk)
                    result[-1] = merged_chunk  # Replace the last chunk
                    i += 1
                else:
                    # Merge with next chunk
                    merged_chunk = self._merge_two_chunks(current_chunk, next_chunk)
                    result.append(merged_chunk)
                    i += 2  # Skip next chunk since we merged it
        
        return result

    def _merge_two_chunks(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two chunks into one."""
        # Combine text
        combined_text = chunk1['text'] + '\n' + chunk2['text']
        
        # Combine positions
        combined_positions = chunk1.get('positions', []) + chunk2.get('positions', [])
        
        # Combine pages
        combined_pages = sorted(list(set(chunk1.get('pages', []) + chunk2.get('pages', []))))
        
        # Determine if spans columns/pages
        spans_columns = len(set([pos.get('column', 'left') for pos in combined_positions])) > 1
        spans_pages = len(combined_pages) > 1
        
        # Create merged chunk
        merged_chunk = {
            'id': f"{chunk1['id']}_merged_{chunk2['id']}",
            'type': chunk1.get('type', 'text'),  # Use first chunk's type
            'text': combined_text,
            'tokens': self._estimate_tokens(combined_text),
            'pages': combined_pages,
            'positions': combined_positions,
            'spans_columns': spans_columns,
            'spans_pages': spans_pages,
            'element_count': chunk1.get('element_count', 1) + chunk2.get('element_count', 1)
        }
        
        return merged_chunk


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


