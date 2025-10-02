import os
import sys
import json
from pathlib import Path
import importlib.util
from collections import defaultdict
import re

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Dynamically import 2_parse_pdf.py
spec = importlib.util.spec_from_file_location("parse_pdf_module", "../../app/2_parse_pdf.py")
parse_pdf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse_pdf_module)

class ImprovedPositionalChunkerV5:
    def __init__(self, pdf_path: Path, output_dir: Path, column_split_x: float = 300.0):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docling_dict_output_path = self.output_dir / "docling_dict_output.json"
        self.docling_markdown_output_path = self.output_dir / "docling_markdown_output.md"
        self.docling_data = self._get_docling_data()
        
        # Filter and sort elements
        self.elements = [
            el for el in self.docling_data.get('texts', [])
            if el.get('prov') and el.get('text', '').strip()
        ]
        # Sort by page, then by reading order (top to bottom, left to right)
        self.elements.sort(key=lambda x: (
            x['prov'][0]['page_no'],
            -x['prov'][0]['bbox']['t'],  # Top to bottom (higher t = higher on page)
            x['prov'][0]['bbox']['l']    # Left to right
        ))
        
        self.markdown_content = self.docling_markdown_output_path.read_text(encoding='utf-8')
        self.column_split_x = column_split_x

    def _get_docling_data(self):
        """Parses PDF with Docling and returns the dictionary output."""
        if not self.docling_dict_output_path.exists():
            print("Converting PDF with Docling...")
            converter = parse_pdf_module.DocumentConverter(format_options={
                parse_pdf_module.InputFormat.PDF: parse_pdf_module.PdfFormatOption(
                    pipeline_options=parse_pdf_module.PdfPipelineOptions()
                )
            })
            result = converter.convert(str(self.pdf_path))
            
            with open(self.docling_markdown_output_path, 'w', encoding='utf-8') as f:
                f.write(result.document.export_to_markdown())

            doc_dict = result.document.export_to_dict()
            with open(self.docling_dict_output_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=2)
            return doc_dict
        else:
            with open(self.docling_dict_output_path, 'r', encoding='utf-8') as f:
                return json.load(f)

    def _get_element_all_bbox_info(self, element):
        """Extracts ALL bounding box and page information from an element's 'prov' data."""
        if 'prov' not in element or not element['prov']:
            return []
        
        bbox_infos = []
        for prov_item in element['prov']:
            if 'bbox' in prov_item:
                bbox = prov_item['bbox']
                page_no = prov_item['page_no']
                
                # Determine column based on bbox center
                center_x = (bbox['l'] + bbox['r']) / 2
                column = "left" if center_x < self.column_split_x else "right"
                
                bbox_infos.append({
                    'page': page_no,
                    'left': bbox['l'],
                    'top': bbox['t'],
                    'right': bbox['r'],
                    'bottom': bbox['b'],
                    'char_span': prov_item.get('charspan'),
                    'column': column
                })
        
        return bbox_infos

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4

    def _create_chunk_with_all_positions(self, elements, chunk_id, chunk_type, page_width):
        """Create a chunk with all position data."""
        chunk_text_parts = []
        all_positions = []
        chunk_pages = set()
        
        for element in elements:
            chunk_text_parts.append(element.get('text', ''))
            bbox_infos = self._get_element_all_bbox_info(element)
            
            for bbox_info in bbox_infos:
                all_positions.append({
                    'page': bbox_info['page'],
                    'column': bbox_info['column'],
                    'bounding_box': {
                        'left': bbox_info['left'],
                        'top': bbox_info['top'],
                        'right': bbox_info['right'],
                        'bottom': bbox_info['bottom']
                    },
                    'char_span': bbox_info['char_span']
                })
                chunk_pages.add(bbox_info['page'])
        
        full_text = "\n".join(chunk_text_parts)
        
        # Determine if chunk spans columns
        columns = set(pos['column'] for pos in all_positions)
        spans_columns = len(columns) > 1
        
        return {
            'id': chunk_id,
            'type': chunk_type,
            'text': full_text,
            'tokens': self._estimate_tokens(full_text),
            'pages': sorted(list(chunk_pages)),
            'positions': all_positions,
            'spans_columns': spans_columns,
            'spans_pages': len(chunk_pages) > 1,
            'char_span': [0, len(full_text)],
            'element_count': len(elements),
            'element_types': list(set(el.get('label', 'text') for el in elements))
        }

    def chunk_column_aware_sections(self, max_tokens: int = 512):
        """
        Chunk by sections but respect column boundaries.
        Elements in different columns are never grouped together.
        """
        chunks = []
        current_section_name = "Introduction"
        current_section_elements = []
        
        for i, element in enumerate(self.elements):
            text = element.get('text', '').strip()
            label = element.get('label', 'text')
            bbox_infos = self._get_element_all_bbox_info(element)
            
            if not text or not bbox_infos:
                continue
            
            # Get the primary position (first one) to determine column
            primary_bbox = bbox_infos[0]
            current_column = primary_bbox['column']
            
            # Check for new section
            is_section_header = label in ['section_header', 'page_header']
            
            if is_section_header:
                # Process previous section elements
                if current_section_elements:
                    # Group elements by column within the section
                    column_groups = defaultdict(list)
                    for el in current_section_elements:
                        el_bbox_infos = self._get_element_all_bbox_info(el)
                        if el_bbox_infos:
                            primary_col = el_bbox_infos[0]['column']
                            column_groups[primary_col].append(el)
                    
                    # Create chunks for each column group
                    for col, col_elements in column_groups.items():
                        chunk_id = f"{current_section_name}_{col}_{len(chunks)}"
                        chunk = self._create_chunk_with_all_positions(
                            col_elements, chunk_id, "column_aware_section", 612.0
                        )
                        chunks.append(chunk)
                
                # Start new section
                current_section_elements = []
                current_section_name = text
            
            # Add element to current section
            current_section_elements.append(element)
            
            # Check if we need to break due to token limit within current column
            current_column_elements = [el for el in current_section_elements 
                                     if self._get_element_all_bbox_info(el) and 
                                     self._get_element_all_bbox_info(el)[0]['column'] == current_column]
            
            total_tokens = sum(self._estimate_tokens(el.get('text', '')) for el in current_column_elements)
            
            if total_tokens > max_tokens and len(current_column_elements) > 1:
                # Split the current column elements
                elements_to_chunk = current_column_elements[:-1]  # All but the last
                chunk_id = f"{current_section_name}_{current_column}_{len(chunks)}"
                chunk = self._create_chunk_with_all_positions(
                    elements_to_chunk, chunk_id, "column_aware_section", 612.0
                )
                chunks.append(chunk)
                
                # Remove chunked elements from current section
                current_section_elements = [el for el in current_section_elements 
                                          if el not in elements_to_chunk]
        
        # Process remaining elements
        if current_section_elements:
            # Group by column
            column_groups = defaultdict(list)
            for el in current_section_elements:
                el_bbox_infos = self._get_element_all_bbox_info(el)
                if el_bbox_infos:
                    primary_col = el_bbox_infos[0]['column']
                    column_groups[primary_col].append(el)
            
            # Create chunks for each column group
            for col, col_elements in column_groups.items():
                chunk_id = f"{current_section_name}_{col}_{len(chunks)}"
                chunk = self._create_chunk_with_all_positions(
                    col_elements, chunk_id, "column_aware_section", 612.0
                )
                chunks.append(chunk)
        
        return chunks

    def run_column_aware_chunking(self):
        """Run the column-aware chunking strategy."""
        print(f"📊 Loaded {len(self.elements)} elements")
        print(f"🔧 Using column split at x={self.column_split_x}")
        
        # Column-aware section chunking
        print("\n=== COLUMN-AWARE SECTION CHUNKING ===")
        chunks = self.chunk_column_aware_sections()
        
        # Statistics
        chunks_with_multiple_pos = sum(1 for c in chunks if len(c['positions']) > 1)
        chunks_spanning_cols = sum(1 for c in chunks if c['spans_columns'])
        chunks_spanning_pages = sum(1 for c in chunks if c['spans_pages'])
        
        print(f"Total chunks: {len(chunks)}")
        print(f"Chunks with multiple positions: {chunks_with_multiple_pos}")
        print(f"Chunks spanning columns: {chunks_spanning_cols}")
        print(f"Chunks spanning pages: {chunks_spanning_pages}")
        print(f"Average tokens per chunk: {sum(c['tokens'] for c in chunks) / len(chunks):.1f}")
        
        # Show sample chunks
        print(f"\nSample chunks:")
        for i, chunk in enumerate(chunks[:5]):
            print(f"  {i+1}. {chunk['id']} ({chunk['tokens']} tokens)")
            print(f"     Pages: {chunk['pages']}, Positions: {len(chunk['positions'])}")
            print(f"     Spans columns: {chunk['spans_columns']}, Spans pages: {chunk['spans_pages']}")
            print(f"     Text: {chunk['text'][:100]}...")
        
        # Save chunks
        output_file = self.output_dir / "improved_chunks_v5_column_aware.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        
        print(f"\n✅ Exported {len(chunks)} chunks to {output_file}")
        
        return chunks


if __name__ == "__main__":
    pdf_file = Path("../../outputs/pdfs/Zj8UqVxClT.pdf")
    output_folder = Path("improved_chunking_output_v5")
    chunker = ImprovedPositionalChunkerV5(pdf_file, output_folder)
    chunker.run_column_aware_chunking()
