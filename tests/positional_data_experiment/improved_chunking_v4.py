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

class ImprovedPositionalChunkerV4:
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

    def _get_element_all_bbox_info(self, element):
        """Extracts ALL bounding box and page number info from an element's 'prov' data."""
        if 'prov' in element and element['prov']:
            bbox_infos = []
            for prov_item in element['prov']:
                bbox = prov_item['bbox']
                page_no = prov_item['page_no']
                bbox_infos.append({
                    'page': page_no,
                    'left': bbox['l'],
                    'top': bbox['t'],
                    'right': bbox['r'],
                    'bottom': bbox['b'],
                    'char_span': prov_item.get('charspan')
                })
            return bbox_infos
        return []

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

    def chunk_by_sections_with_all_positions(self, max_tokens: int = 512, page_width: float = 612.0):
        """
        Chunk by sections, capturing ALL position data for elements that span multiple columns/pages.
        """
        chunks = []
        current_section_elements = []
        current_section_name = "Introduction"
        
        for i, element in enumerate(self.elements):
            text = element.get('text', '').strip()
            label = element.get('label', 'text')
            all_bbox_infos = self._get_element_all_bbox_info(element)
            
            if not text or not all_bbox_infos:
                continue
            
            # Check for new section
            is_section_header = label in ['section_header', 'page_header']
            
            if is_section_header:
                # Process previous section if it exists
                if current_section_elements:
                    chunk = self._create_section_chunk_with_all_positions(
                        current_section_elements, 
                        current_section_name, 
                        len(chunks),
                        page_width
                    )
                    chunks.append(chunk)
                
                # Start new section
                current_section_elements = []
                current_section_name = text
            
            # Add element to current section
            current_section_elements.append(element)
            
            # Check if we need to break due to token limit
            total_tokens = sum(self._estimate_tokens(el.get('text', '')) for el in current_section_elements)
            if total_tokens > max_tokens and len(current_section_elements) > 1:
                # Split the section into multiple chunks
                chunk = self._create_section_chunk_with_all_positions(
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
            chunk = self._create_section_chunk_with_all_positions(
                current_section_elements, 
                current_section_name, 
                len(chunks),
                page_width
            )
            chunks.append(chunk)
        
        return chunks

    def _create_section_chunk_with_all_positions(self, elements, section_name, chunk_index, page_width):
        """Create a chunk with ALL position data for elements spanning multiple columns/pages."""
        texts = []
        positions = []  # List of position data for different parts of the section
        pages = set()
        char_spans = []
        
        # Process each element and ALL its position data
        for element in elements:
            all_bbox_infos = self._get_element_all_bbox_info(element)
            text = element.get('text', '').strip()
            
            if text:
                texts.append(text)
            
            # Create position data for EACH bbox info (each column/page part)
            for bbox_info in all_bbox_infos:
                page = bbox_info['page']
                column = self._determine_column(bbox_info, page_width)
                
                # Check if we already have a position for this page/column combination
                existing_pos = None
                for pos in positions:
                    if pos['page'] == page and pos['column'] == column:
                        existing_pos = pos
                        break
                
                if existing_pos:
                    # Merge with existing position
                    existing_pos['bounding_box']['left'] = min(existing_pos['bounding_box']['left'], bbox_info['left'])
                    existing_pos['bounding_box']['top'] = max(existing_pos['bounding_box']['top'], bbox_info['top'])
                    existing_pos['bounding_box']['right'] = max(existing_pos['bounding_box']['right'], bbox_info['right'])
                    existing_pos['bounding_box']['bottom'] = min(existing_pos['bounding_box']['bottom'], bbox_info['bottom'])
                    existing_pos['element_count'] += 1
                    if bbox_info['char_span']:
                        char_spans.append(bbox_info['char_span'])
                else:
                    # Create new position
                    positions.append({
                        'page': page,
                        'column': column,
                        'bounding_box': {
                            'left': bbox_info['left'],
                            'top': bbox_info['top'],
                            'right': bbox_info['right'],
                            'bottom': bbox_info['bottom']
                        },
                        'text': text,  # Full text for this position
                        'element_count': 1,
                        'char_span': bbox_info['char_span']
                    })
                    
                    pages.add(page)
                    if bbox_info['char_span']:
                        char_spans.append(bbox_info['char_span'])
        
        full_text = "\n".join(texts)
        
        # Create chunk ID
        section_clean = re.sub(r'[^\w\s-]', '', section_name)[:20].strip()
        chunk_id = f"{section_clean}_{chunk_index}" if section_clean else f"chunk_{chunk_index}"
        
        return {
            'id': chunk_id,
            'type': 'section_all_positions',
            'section': section_name,
            'text': full_text,
            'tokens': self._estimate_tokens(full_text),
            'pages': sorted(list(pages)),
            'positions': positions,  # ALL position entries
            'char_span': char_spans[0] if char_spans else [0, len(full_text)],
            'element_count': len(elements),
            'element_types': list(set(el.get('label', 'text') for el in elements)),
            'spans_columns': len(set(pos['column'] for pos in positions)) > 1,
            'spans_pages': len(pages) > 1
        }

    def run_improved_chunking_v4(self):
        """Run the improved chunking strategies."""
        print(f"📊 Loaded {len(self.elements)} elements")
        
        # Section-based chunking with ALL positions
        print("\n=== SECTION-BASED CHUNKING WITH ALL POSITIONS ===")
        section_chunks = self.chunk_by_sections_with_all_positions(max_tokens=512, page_width=612.0)
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
        print(f"\nSample chunks with ALL position details:")
        for i, chunk in enumerate(section_chunks[:3]):
            print(f"  {i+1}. {chunk['id']} ({chunk['tokens']} tokens)")
            print(f"     Positions: {len(chunk['positions'])}")
            for j, pos in enumerate(chunk['positions']):
                print(f"       {j+1}. Page {pos['page']}, {pos['column']} column")
                print(f"          BBox: ({pos['bounding_box']['left']:.1f}, {pos['bounding_box']['top']:.1f}, {pos['bounding_box']['right']:.1f}, {pos['bounding_box']['bottom']:.1f})")
                if 'char_span' in pos and pos['char_span']:
                    print(f"          Char span: {pos['char_span']}")
            print(f"     Text: {chunk['text'][:80]}...")
        
        with open(self.output_dir / "improved_chunks_v4_all_positions.json", 'w', encoding='utf-8') as f:
            json.dump(section_chunks, f, indent=2)
        print(f"✅ Exported {len(section_chunks)} chunks to improved_chunks_v4_all_positions.json")
        
        print("\n🎯 IMPROVEMENTS:")
        print("1. ✅ Captures ALL position data from Docling elements")
        print("2. ✅ Handles elements with multiple prov entries")
        print("3. ✅ Properly maps text that spans columns")
        print("4. ✅ Maintains section coherence")
        print("5. ✅ Perfect for frontend highlighting with complete coverage")


if __name__ == "__main__":
    pdf_file = Path("../../outputs/pdfs/Zj8UqVxClT.pdf")
    output_folder = Path("improved_chunking_output_v4")
    chunker = ImprovedPositionalChunkerV4(pdf_file, output_folder)
    chunker.run_improved_chunking_v4()
