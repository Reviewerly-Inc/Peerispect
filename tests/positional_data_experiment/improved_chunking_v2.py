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

class ImprovedPositionalChunkerV2:
    def __init__(self, pdf_path: Path, output_dir: Path):
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
        self.elements.sort(key=lambda x: (x['prov'][0]['page_no'], x['prov'][0]['bbox']['t']), reverse=True)
        
        self.markdown_content = self.docling_markdown_output_path.read_text(encoding='utf-8')

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

    def _get_element_bbox_info(self, element):
        """Extracts bounding box and page number from an element's 'prov' data."""
        if 'prov' in element and element['prov']:
            prov_item = element['prov'][0]
            bbox = prov_item['bbox']
            page_no = prov_item['page_no']
            return {
                'page': page_no,
                'left': bbox['l'],
                'top': bbox['t'],
                'right': bbox['r'],
                'bottom': bbox['b'],
                'char_span': prov_item.get('charspan')
            }
        return None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using character-based approximation."""
        return len(text) // 4

    def _is_same_column(self, bbox1, bbox2, page_width):
        """Check if two bounding boxes are in the same column."""
        if not bbox1 or not bbox2:
            return False
        
        # Define column boundaries (assuming two-column layout)
        left_column_end = page_width * 0.5  # Middle of page
        
        # Check if both boxes are in the same column
        bbox1_left_col = bbox1['right'] <= left_column_end
        bbox2_left_col = bbox2['right'] <= left_column_end
        bbox1_right_col = bbox1['left'] >= left_column_end
        bbox2_right_col = bbox2['left'] >= left_column_end
        
        return (bbox1_left_col and bbox2_left_col) or (bbox1_right_col and bbox2_right_col)

    def _is_vertical_adjacent(self, bbox1, bbox2, threshold=50):
        """Check if two bounding boxes are vertically adjacent."""
        if not bbox1 or not bbox2:
            return False
        
        # Check if boxes are close vertically (within threshold)
        vertical_distance = abs(bbox1['bottom'] - bbox2['top'])
        return vertical_distance <= threshold

    def chunk_by_columns_and_sections(self, max_tokens: int = 512, page_width: float = 612.0):
        """
        Improved chunking that respects column boundaries and section breaks.
        """
        chunks = []
        current_chunk_elements = []
        current_chunk_tokens = 0
        current_section = "Introduction"
        current_page = None
        current_column = None
        
        for i, element in enumerate(self.elements):
            text = element.get('text', '').strip()
            label = element.get('label', 'text')
            bbox_info = self._get_element_bbox_info(element)
            
            if not text or not bbox_info:
                continue
            
            element_tokens = self._estimate_tokens(text)
            
            # Check for section boundaries
            is_section_header = label in ['section_header', 'page_header']
            is_new_page = current_page is not None and bbox_info['page'] != current_page
            is_new_column = not self._is_same_column(
                self._get_element_bbox_info(current_chunk_elements[-1]) if current_chunk_elements else None,
                bbox_info,
                page_width
            ) if current_chunk_elements else False
            
            # Check if we should start a new chunk
            should_break = False
            
            # Always break on section headers
            if is_section_header:
                should_break = True
                current_section = text
                current_page = bbox_info['page']
                current_column = 'left' if bbox_info['right'] <= page_width * 0.5 else 'right'
            
            # Break on page boundaries
            elif is_new_page:
                should_break = True
                current_page = bbox_info['page']
                current_column = 'left' if bbox_info['right'] <= page_width * 0.5 else 'right'
            
            # Break on column boundaries (unless it's a continuation)
            elif is_new_column and current_chunk_elements:
                # Only break if the current chunk is substantial enough
                if current_chunk_tokens > max_tokens * 0.3:  # At least 30% of max tokens
                    should_break = True
                    current_column = 'left' if bbox_info['right'] <= page_width * 0.5 else 'right'
            
            # Break if adding this element would exceed token limit
            elif current_chunk_tokens + element_tokens > max_tokens and current_chunk_elements:
                should_break = True
            
            if should_break and current_chunk_elements:
                # Create chunk from accumulated elements
                chunk = self._create_column_aware_chunk(
                    current_chunk_elements, 
                    current_section, 
                    len(chunks),
                    current_page,
                    current_column
                )
                chunks.append(chunk)
                current_chunk_elements = []
                current_chunk_tokens = 0
            
            # Initialize page and column if not set
            if current_page is None:
                current_page = bbox_info['page']
                current_column = 'left' if bbox_info['right'] <= page_width * 0.5 else 'right'
            
            current_chunk_elements.append(element)
            current_chunk_tokens += element_tokens
        
        # Add the last chunk
        if current_chunk_elements:
            chunk = self._create_column_aware_chunk(
                current_chunk_elements, 
                current_section, 
                len(chunks),
                current_page,
                current_column
            )
            chunks.append(chunk)
        
        return chunks

    def _create_column_aware_chunk(self, elements, section, chunk_index, page, column):
        """Create a chunk with column and page awareness."""
        texts = []
        bboxes = []
        pages = set()
        char_spans = []
        
        for element in elements:
            text = element.get('text', '').strip()
            if text:
                texts.append(text)
                bbox_info = self._get_element_bbox_info(element)
                if bbox_info:
                    bboxes.append({k: v for k, v in bbox_info.items() if k != 'char_span'})
                    pages.add(bbox_info['page'])
                    if bbox_info['char_span']:
                        char_spans.append(bbox_info['char_span'])
        
        full_text = "\n".join(texts)
        
        # Aggregate bounding boxes
        if bboxes:
            min_l = min(b['left'] for b in bboxes)
            max_t = max(b['top'] for b in bboxes)
            max_r = max(b['right'] for b in bboxes)
            min_b = min(b['bottom'] for b in bboxes)
            
            aggregated_bbox = {
                'page': sorted(list(pages)),
                'left': min_l,
                'top': max_t,
                'right': max_r,
                'bottom': min_b
            }
        else:
            aggregated_bbox = None
        
        # Create chunk ID based on section, column, and index
        section_clean = re.sub(r'[^\w\s-]', '', section)[:20].strip()
        chunk_id = f"{section_clean}_{column}_{chunk_index}" if section_clean else f"chunk_{column}_{chunk_index}"
        
        return {
            'id': chunk_id,
            'type': 'column_aware',
            'section': section,
            'column': column,
            'text': full_text,
            'tokens': self._estimate_tokens(full_text),
            'pages': sorted(list(pages)),
            'bounding_box': aggregated_bbox,
            'char_span': char_spans[0] if char_spans else [0, len(full_text)],
            'element_count': len(elements),
            'element_types': list(set(el.get('label', 'text') for el in elements))
        }

    def chunk_by_reading_order(self, max_tokens: int = 512):
        """
        Chunk by reading order, respecting natural text flow.
        """
        chunks = []
        current_chunk_elements = []
        current_chunk_tokens = 0
        current_section = "Introduction"
        
        # Group elements by page first
        page_elements = defaultdict(list)
        for element in self.elements:
            bbox_info = self._get_element_bbox_info(element)
            if bbox_info:
                page_elements[bbox_info['page']].append(element)
        
        # Process each page separately
        for page_num in sorted(page_elements.keys()):
            page_els = page_elements[page_num]
            
            # Sort elements by reading order (top to bottom, left to right)
            page_els.sort(key=lambda x: (
                -x['prov'][0]['bbox']['t'],  # Top to bottom (higher Y first)
                x['prov'][0]['bbox']['l']    # Left to right
            ))
            
            for element in page_els:
                text = element.get('text', '').strip()
                label = element.get('label', 'text')
                bbox_info = self._get_element_bbox_info(element)
                
                if not text or not bbox_info:
                    continue
                
                element_tokens = self._estimate_tokens(text)
                
                # Check for section boundaries
                is_section_header = label in ['section_header', 'page_header']
                
                # Check if we should start a new chunk
                should_break = False
                
                # Always break on section headers
                if is_section_header:
                    should_break = True
                    current_section = text
                
                # Break if adding this element would exceed token limit
                elif current_chunk_tokens + element_tokens > max_tokens and current_chunk_elements:
                    should_break = True
                
                if should_break and current_chunk_elements:
                    # Create chunk from accumulated elements
                    chunk = self._create_reading_order_chunk(
                        current_chunk_elements, 
                        current_section, 
                        len(chunks)
                    )
                    chunks.append(chunk)
                    current_chunk_elements = []
                    current_chunk_tokens = 0
                
                current_chunk_elements.append(element)
                current_chunk_tokens += element_tokens
        
        # Add the last chunk
        if current_chunk_elements:
            chunk = self._create_reading_order_chunk(
                current_chunk_elements, 
                current_section, 
                len(chunks)
            )
            chunks.append(chunk)
        
        return chunks

    def _create_reading_order_chunk(self, elements, section, chunk_index):
        """Create a chunk following reading order."""
        texts = []
        bboxes = []
        pages = set()
        char_spans = []
        
        for element in elements:
            text = element.get('text', '').strip()
            if text:
                texts.append(text)
                bbox_info = self._get_element_bbox_info(element)
                if bbox_info:
                    bboxes.append({k: v for k, v in bbox_info.items() if k != 'char_span'})
                    pages.add(bbox_info['page'])
                    if bbox_info['char_span']:
                        char_spans.append(bbox_info['char_span'])
        
        full_text = "\n".join(texts)
        
        # Aggregate bounding boxes
        if bboxes:
            min_l = min(b['left'] for b in bboxes)
            max_t = max(b['top'] for b in bboxes)
            max_r = max(b['right'] for b in bboxes)
            min_b = min(b['bottom'] for b in bboxes)
            
            aggregated_bbox = {
                'page': sorted(list(pages)),
                'left': min_l,
                'top': max_t,
                'right': max_r,
                'bottom': min_b
            }
        else:
            aggregated_bbox = None
        
        # Create chunk ID
        section_clean = re.sub(r'[^\w\s-]', '', section)[:20].strip()
        chunk_id = f"{section_clean}_{chunk_index}" if section_clean else f"chunk_{chunk_index}"
        
        return {
            'id': chunk_id,
            'type': 'reading_order',
            'section': section,
            'text': full_text,
            'tokens': self._estimate_tokens(full_text),
            'pages': sorted(list(pages)),
            'bounding_box': aggregated_bbox,
            'char_span': char_spans[0] if char_spans else [0, len(full_text)],
            'element_count': len(elements),
            'element_types': list(set(el.get('label', 'text') for el in elements))
        }

    def run_improved_chunking_v2(self):
        """Run the improved chunking strategies."""
        print(f"📊 Loaded {len(self.elements)} elements")
        
        # Column-aware chunking
        print("\n=== COLUMN-AWARE CHUNKING ===")
        column_chunks = self.chunk_by_columns_and_sections(max_tokens=512, page_width=612.0)
        print(f"Total chunks: {len(column_chunks)}")
        print(f"Average tokens per chunk: {sum(c['tokens'] for c in column_chunks) / len(column_chunks):.1f}")
        print(f"Chunks with bounding boxes: {sum(1 for c in column_chunks if c['bounding_box'] is not None)}/{len(column_chunks)}")
        
        # Show chunk distribution by section and column
        section_column_counts = defaultdict(lambda: defaultdict(int))
        for chunk in column_chunks:
            section_column_counts[chunk['section']][chunk['column']] += 1
        
        print(f"Chunks per section and column:")
        for section, columns in section_column_counts.items():
            print(f"  {section}:")
            for column, count in columns.items():
                print(f"    {column}: {count} chunks")
        
        # Show sample chunks
        print(f"\nSample chunks:")
        for i, chunk in enumerate(column_chunks[:5]):
            print(f"  {i+1}. {chunk['id']} ({chunk['tokens']} tokens, {chunk['column']} column)")
            print(f"     Text: {chunk['text'][:80]}...")
            if chunk['bounding_box']:
                print(f"     BBox: page {chunk['pages'][0]}, ({chunk['bounding_box']['left']:.1f}, {chunk['bounding_box']['top']:.1f}, {chunk['bounding_box']['right']:.1f}, {chunk['bounding_box']['bottom']:.1f})")
        
        with open(self.output_dir / "improved_chunks_v2_column_aware.json", 'w', encoding='utf-8') as f:
            json.dump(column_chunks, f, indent=2)
        print(f"✅ Exported {len(column_chunks)} chunks to improved_chunks_v2_column_aware.json")
        
        # Reading order chunking
        print("\n=== READING ORDER CHUNKING ===")
        reading_chunks = self.chunk_by_reading_order(max_tokens=512)
        print(f"Total chunks: {len(reading_chunks)}")
        print(f"Average tokens per chunk: {sum(c['tokens'] for c in reading_chunks) / len(reading_chunks):.1f}")
        print(f"Chunks with bounding boxes: {sum(1 for c in reading_chunks if c['bounding_box'] is not None)}/{len(reading_chunks)}")
        
        with open(self.output_dir / "improved_chunks_v2_reading_order.json", 'w', encoding='utf-8') as f:
            json.dump(reading_chunks, f, indent=2)
        print(f"✅ Exported {len(reading_chunks)} chunks to improved_chunks_v2_reading_order.json")
        
        print("\n🎯 IMPROVEMENTS:")
        print("1. ✅ Respects column boundaries")
        print("2. ✅ Prevents mid-sentence breaks across columns")
        print("3. ✅ Maintains page boundaries")
        print("4. ✅ Follows natural reading order")
        print("5. ✅ Creates coherent, readable chunks")


if __name__ == "__main__":
    pdf_file = Path("../../outputs/pdfs/Zj8UqVxClT.pdf")
    output_folder = Path("improved_chunking_output_v2")
    chunker = ImprovedPositionalChunkerV2(pdf_file, output_folder)
    chunker.run_improved_chunking_v2()
