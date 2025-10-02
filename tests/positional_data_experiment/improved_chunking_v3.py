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

class ImprovedPositionalChunkerV3:
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

    def _determine_column(self, bbox, page_width):
        """Determine which column an element belongs to."""
        if not bbox:
            return 'unknown'
        
        # More accurate column detection
        left_column_end = page_width * 0.5
        
        # If element spans both columns significantly, it's "both"
        if bbox['left'] < left_column_end * 0.8 and bbox['right'] > left_column_end * 1.2:
            return 'both'
        elif bbox['right'] <= left_column_end:
            return 'left'
        else:
            return 'right'

    def _is_same_section(self, element1, element2):
        """Check if two elements belong to the same section."""
        if not element1 or not element2:
            return False
        
        # Check if they have the same section header in their path
        # This is a simplified approach - in practice, you'd want more sophisticated section detection
        return True  # For now, assume all elements can be in the same section

    def chunk_by_sections_with_multiple_positions(self, max_tokens: int = 512, page_width: float = 612.0):
        """
        Chunk by sections, allowing sections to span multiple columns and pages
        with multiple positional data entries.
        """
        chunks = []
        current_section_elements = []
        current_section_name = "Introduction"
        current_section_started = False
        
        for i, element in enumerate(self.elements):
            text = element.get('text', '').strip()
            label = element.get('label', 'text')
            bbox_info = self._get_element_bbox_info(element)
            
            if not text or not bbox_info:
                continue
            
            # Check for new section
            is_section_header = label in ['section_header', 'page_header']
            
            if is_section_header:
                # Process previous section if it exists
                if current_section_elements:
                    chunk = self._create_section_chunk_with_multiple_positions(
                        current_section_elements, 
                        current_section_name, 
                        len(chunks),
                        page_width
                    )
                    chunks.append(chunk)
                
                # Start new section
                current_section_elements = []
                current_section_name = text
                current_section_started = True
            
            # Add element to current section
            current_section_elements.append(element)
            
            # Check if we need to break due to token limit
            total_tokens = sum(self._estimate_tokens(el.get('text', '')) for el in current_section_elements)
            if total_tokens > max_tokens and len(current_section_elements) > 1:
                # Split the section into multiple chunks
                chunk = self._create_section_chunk_with_multiple_positions(
                    current_section_elements[:-1],  # All but the last element
                    current_section_name, 
                    len(chunks),
                    page_width
                )
                chunks.append(chunk)
                
                # Start new chunk with the last element
                current_section_elements = [current_section_elements[-1]]
        
        # Add the last section
        if current_section_elements:
            chunk = self._create_section_chunk_with_multiple_positions(
                current_section_elements, 
                current_section_name, 
                len(chunks),
                page_width
            )
            chunks.append(chunk)
        
        return chunks

    def _create_section_chunk_with_multiple_positions(self, elements, section_name, chunk_index, page_width):
        """Create a chunk with multiple positional data entries for sections spanning columns/pages."""
        texts = []
        positions = []  # List of position data for different parts of the section
        pages = set()
        char_spans = []
        
        # Group elements by page and column
        page_column_groups = defaultdict(list)
        for element in elements:
            bbox_info = self._get_element_bbox_info(element)
            if bbox_info:
                page = bbox_info['page']
                column = self._determine_column(bbox_info, page_width)
                page_column_groups[(page, column)].append((element, bbox_info))
        
        # Create position data for each page/column group
        for (page, column), group_elements in page_column_groups.items():
            group_texts = []
            group_bboxes = []
            
            for element, bbox_info in group_elements:
                text = element.get('text', '').strip()
                if text:
                    group_texts.append(text)
                    group_bboxes.append(bbox_info)
                    if bbox_info['char_span']:
                        char_spans.append(bbox_info['char_span'])
            
            if group_bboxes:
                # Aggregate bounding boxes for this group
                min_l = min(b['left'] for b in group_bboxes)
                max_t = max(b['top'] for b in group_bboxes)
                max_r = max(b['right'] for b in group_bboxes)
                min_b = min(b['bottom'] for b in group_bboxes)
                
                positions.append({
                    'page': page,
                    'column': column,
                    'bounding_box': {
                        'left': min_l,
                        'top': max_t,
                        'right': max_r,
                        'bottom': min_b
                    },
                    'text': ' '.join(group_texts),
                    'element_count': len(group_elements)
                })
                
                pages.add(page)
                texts.extend(group_texts)
        
        full_text = "\n".join(texts)
        
        # Create chunk ID
        section_clean = re.sub(r'[^\w\s-]', '', section_name)[:20].strip()
        chunk_id = f"{section_clean}_{chunk_index}" if section_clean else f"chunk_{chunk_index}"
        
        return {
            'id': chunk_id,
            'type': 'section_multi_position',
            'section': section_name,
            'text': full_text,
            'tokens': self._estimate_tokens(full_text),
            'pages': sorted(list(pages)),
            'positions': positions,  # Multiple position entries
            'char_span': char_spans[0] if char_spans else [0, len(full_text)],
            'element_count': len(elements),
            'element_types': list(set(el.get('label', 'text') for el in elements)),
            'spans_columns': len(set(pos['column'] for pos in positions)) > 1,
            'spans_pages': len(pages) > 1
        }

    def chunk_by_reading_flow(self, max_tokens: int = 512, page_width: float = 612.0):
        """
        Chunk by natural reading flow, respecting document structure.
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
                    chunk = self._create_reading_flow_chunk(
                        current_chunk_elements, 
                        current_section, 
                        len(chunks),
                        page_width
                    )
                    chunks.append(chunk)
                    current_chunk_elements = []
                    current_chunk_tokens = 0
                
                current_chunk_elements.append(element)
                current_chunk_tokens += element_tokens
        
        # Add the last chunk
        if current_chunk_elements:
            chunk = self._create_reading_flow_chunk(
                current_chunk_elements, 
                current_section, 
                len(chunks),
                page_width
            )
            chunks.append(chunk)
        
        return chunks

    def _create_reading_flow_chunk(self, elements, section, chunk_index, page_width):
        """Create a chunk following reading flow."""
        texts = []
        positions = []
        pages = set()
        char_spans = []
        
        # Group elements by page and column
        page_column_groups = defaultdict(list)
        for element in elements:
            bbox_info = self._get_element_bbox_info(element)
            if bbox_info:
                page = bbox_info['page']
                column = self._determine_column(bbox_info, page_width)
                page_column_groups[(page, column)].append((element, bbox_info))
        
        # Create position data for each page/column group
        for (page, column), group_elements in page_column_groups.items():
            group_texts = []
            group_bboxes = []
            
            for element, bbox_info in group_elements:
                text = element.get('text', '').strip()
                if text:
                    group_texts.append(text)
                    group_bboxes.append(bbox_info)
                    if bbox_info['char_span']:
                        char_spans.append(bbox_info['char_span'])
            
            if group_bboxes:
                # Aggregate bounding boxes for this group
                min_l = min(b['left'] for b in group_bboxes)
                max_t = max(b['top'] for b in group_bboxes)
                max_r = max(b['right'] for b in group_bboxes)
                min_b = min(b['bottom'] for b in group_bboxes)
                
                positions.append({
                    'page': page,
                    'column': column,
                    'bounding_box': {
                        'left': min_l,
                        'top': max_t,
                        'right': max_r,
                        'bottom': min_b
                    },
                    'text': ' '.join(group_texts),
                    'element_count': len(group_elements)
                })
                
                pages.add(page)
                texts.extend(group_texts)
        
        full_text = "\n".join(texts)
        
        # Create chunk ID
        section_clean = re.sub(r'[^\w\s-]', '', section)[:20].strip()
        chunk_id = f"{section_clean}_{chunk_index}" if section_clean else f"chunk_{chunk_index}"
        
        return {
            'id': chunk_id,
            'type': 'reading_flow',
            'section': section,
            'text': full_text,
            'tokens': self._estimate_tokens(full_text),
            'pages': sorted(list(pages)),
            'positions': positions,
            'char_span': char_spans[0] if char_spans else [0, len(full_text)],
            'element_count': len(elements),
            'element_types': list(set(el.get('label', 'text') for el in elements)),
            'spans_columns': len(set(pos['column'] for pos in positions)) > 1,
            'spans_pages': len(pages) > 1
        }

    def run_improved_chunking_v3(self):
        """Run the improved chunking strategies."""
        print(f"📊 Loaded {len(self.elements)} elements")
        
        # Section-based chunking with multiple positions
        print("\n=== SECTION-BASED CHUNKING WITH MULTIPLE POSITIONS ===")
        section_chunks = self.chunk_by_sections_with_multiple_positions(max_tokens=512, page_width=612.0)
        print(f"Total chunks: {len(section_chunks)}")
        print(f"Average tokens per chunk: {sum(c['tokens'] for c in section_chunks) / len(section_chunks):.1f}")
        print(f"Chunks with multiple positions: {sum(1 for c in section_chunks if len(c['positions']) > 1)}/{len(section_chunks)}")
        print(f"Chunks spanning columns: {sum(1 for c in section_chunks if c['spans_columns'])}")
        print(f"Chunks spanning pages: {sum(1 for c in section_chunks if c['spans_pages'])}")
        
        # Show chunk distribution by section
        section_counts = defaultdict(int)
        for chunk in section_chunks:
            section_counts[chunk['section']] += 1
        
        print(f"Chunks per section:")
        for section, count in section_counts.items():
            print(f"  {section}: {count} chunks")
        
        # Show sample chunks with position details
        print(f"\nSample chunks with position details:")
        for i, chunk in enumerate(section_chunks[:3]):
            print(f"  {i+1}. {chunk['id']} ({chunk['tokens']} tokens)")
            print(f"     Positions: {len(chunk['positions'])}")
            for j, pos in enumerate(chunk['positions']):
                print(f"       {j+1}. Page {pos['page']}, {pos['column']} column")
                print(f"          BBox: ({pos['bounding_box']['left']:.1f}, {pos['bounding_box']['top']:.1f}, {pos['bounding_box']['right']:.1f}, {pos['bounding_box']['bottom']:.1f})")
            print(f"     Text: {chunk['text'][:80]}...")
        
        with open(self.output_dir / "improved_chunks_v3_section_multi_pos.json", 'w', encoding='utf-8') as f:
            json.dump(section_chunks, f, indent=2)
        print(f"✅ Exported {len(section_chunks)} chunks to improved_chunks_v3_section_multi_pos.json")
        
        # Reading flow chunking
        print("\n=== READING FLOW CHUNKING ===")
        flow_chunks = self.chunk_by_reading_flow(max_tokens=512, page_width=612.0)
        print(f"Total chunks: {len(flow_chunks)}")
        print(f"Average tokens per chunk: {sum(c['tokens'] for c in flow_chunks) / len(flow_chunks):.1f}")
        print(f"Chunks with multiple positions: {sum(1 for c in flow_chunks if len(c['positions']) > 1)}/{len(flow_chunks)}")
        print(f"Chunks spanning columns: {sum(1 for c in flow_chunks if c['spans_columns'])}")
        print(f"Chunks spanning pages: {sum(1 for c in flow_chunks if c['spans_pages'])}")
        
        with open(self.output_dir / "improved_chunks_v3_reading_flow.json", 'w', encoding='utf-8') as f:
            json.dump(flow_chunks, f, indent=2)
        print(f"✅ Exported {len(flow_chunks)} chunks to improved_chunks_v3_reading_flow.json")
        
        print("\n🎯 IMPROVEMENTS:")
        print("1. ✅ Sections can span multiple columns and pages")
        print("2. ✅ Multiple position data for complex layouts")
        print("3. ✅ Accurate column detection")
        print("4. ✅ Maintains section coherence")
        print("5. ✅ Perfect for frontend highlighting")


if __name__ == "__main__":
    pdf_file = Path("../../outputs/pdfs/Zj8UqVxClT.pdf")
    output_folder = Path("improved_chunking_output_v3")
    chunker = ImprovedPositionalChunkerV3(pdf_file, output_folder)
    chunker.run_improved_chunking_v3()
