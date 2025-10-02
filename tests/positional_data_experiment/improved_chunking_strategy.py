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

class ImprovedPositionalChunker:
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

    def _aggregate_bboxes(self, bboxes: list) -> dict:
        """Aggregates multiple bounding boxes into a single encompassing box."""
        if not bboxes:
            return None
        
        min_l = min(b['left'] for b in bboxes)
        max_t = max(b['top'] for b in bboxes)
        max_r = max(b['right'] for b in bboxes)
        min_b = min(b['bottom'] for b in bboxes)
        
        pages = sorted(list(set(b['page'] for b in bboxes)))
        return {
            'page': pages,
            'left': min_l,
            'top': max_t,
            'right': max_r,
            'bottom': min_b
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using character-based approximation."""
        return len(text) // 4

    def chunk_smart_hybrid(self, target_chunks: int = 30, max_tokens: int = 512) -> list:
        """
        Smart hybrid chunking that aims for a target number of chunks while maintaining
        semantic coherence and positional accuracy.
        
        Strategy:
        1. Identify major sections (headers, page breaks)
        2. Group elements within sections by proximity and content type
        3. Respect token limits while maintaining readability
        4. Ensure each chunk has accurate positional data
        """
        chunks = []
        current_chunk_elements = []
        current_chunk_tokens = 0
        current_section = "Introduction"
        
        for i, element in enumerate(self.elements):
            text = element.get('text', '').strip()
            label = element.get('label', 'text')
            bbox_info = self._get_element_bbox_info(element)
            
            if not text or not bbox_info:
                continue
            
            element_tokens = self._estimate_tokens(text)
            
            # Check for section boundaries
            is_section_header = label in ['section_header', 'page_header']
            is_new_page = False
            if current_chunk_elements:
                last_bbox = self._get_element_bbox_info(current_chunk_elements[-1])
                if last_bbox and bbox_info['page'] != last_bbox['page']:
                    is_new_page = True
            
            # Check if we should start a new chunk
            should_break = False
            
            # Always break on section headers
            if is_section_header:
                should_break = True
                current_section = text
            
            # Break if adding this element would exceed token limit
            elif current_chunk_tokens + element_tokens > max_tokens and current_chunk_elements:
                should_break = True
            
            # Break on page boundaries if chunk is getting large
            elif is_new_page and current_chunk_tokens > max_tokens * 0.7:
                should_break = True
            
            # Break if we have too many chunks and need to consolidate
            elif len(chunks) >= target_chunks and current_chunk_tokens > max_tokens * 0.5:
                should_break = True
            
            if should_break and current_chunk_elements:
                # Create chunk from accumulated elements
                chunk = self._create_smart_chunk(current_chunk_elements, current_section, len(chunks))
                chunks.append(chunk)
                current_chunk_elements = []
                current_chunk_tokens = 0
            
            current_chunk_elements.append(element)
            current_chunk_tokens += element_tokens
        
        # Add the last chunk
        if current_chunk_elements:
            chunk = self._create_smart_chunk(current_chunk_elements, current_section, len(chunks))
            chunks.append(chunk)
        
        return chunks

    def _create_smart_chunk(self, elements: list, section: str, chunk_index: int) -> dict:
        """Create a smart chunk with metadata and positional data."""
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
        aggregated_bbox = self._aggregate_bboxes(bboxes)
        
        # Create chunk ID based on section and index
        section_clean = re.sub(r'[^\w\s-]', '', section)[:30].strip()
        chunk_id = f"{section_clean}_{chunk_index}" if section_clean else f"chunk_{chunk_index}"
        
        return {
            'id': chunk_id,
            'type': 'smart_hybrid',
            'section': section,
            'text': full_text,
            'tokens': self._estimate_tokens(full_text),
            'pages': sorted(list(pages)),
            'bounding_box': aggregated_bbox,
            'char_span': char_spans[0] if char_spans else [0, len(full_text)],
            'element_count': len(elements),
            'element_types': list(set(el.get('label', 'text') for el in elements))
        }

    def chunk_adaptive_sections(self, max_tokens: int = 512) -> list:
        """
        Adaptive section-based chunking that splits large sections intelligently.
        """
        chunks = []
        current_section_elements = []
        current_section_name = "Introduction"
        
        for element in self.elements:
            label = element.get('label', 'text')
            text = element.get('text', '').strip()
            
            if not text:
                continue
            
            # New section detected
            if label in ['section_header', 'page_header'] and current_section_elements:
                # Process current section
                section_chunks = self._split_section_if_needed(
                    current_section_elements, 
                    current_section_name, 
                    max_tokens
                )
                chunks.extend(section_chunks)
                
                # Start new section
                current_section_elements = []
                current_section_name = text
            
            current_section_elements.append(element)
        
        # Process final section
        if current_section_elements:
            section_chunks = self._split_section_if_needed(
                current_section_elements, 
                current_section_name, 
                max_tokens
            )
            chunks.extend(section_chunks)
        
        return chunks

    def _split_section_if_needed(self, elements: list, section_name: str, max_tokens: int) -> list:
        """Split a section into multiple chunks if it's too large."""
        chunks = []
        current_chunk_elements = []
        current_tokens = 0
        
        for element in elements:
            text = element.get('text', '').strip()
            if not text:
                continue
            
            element_tokens = self._estimate_tokens(text)
            
            # If adding this element would exceed limit, start new chunk
            if current_tokens + element_tokens > max_tokens and current_chunk_elements:
                chunk = self._create_smart_chunk(current_chunk_elements, section_name, len(chunks))
                chunks.append(chunk)
                current_chunk_elements = []
                current_tokens = 0
            
            current_chunk_elements.append(element)
            current_tokens += element_tokens
        
        # Add final chunk
        if current_chunk_elements:
            chunk = self._create_smart_chunk(current_chunk_elements, section_name, len(chunks))
            chunks.append(chunk)
        
        return chunks

    def run_improved_chunking(self):
        """Run the improved chunking strategies."""
        print(f"📊 Loaded {len(self.elements)} elements")
        
        # Smart hybrid chunking (target ~30 chunks)
        print("\n=== SMART HYBRID CHUNKING (Target: ~30 chunks) ===")
        smart_chunks = self.chunk_smart_hybrid(target_chunks=30, max_tokens=512)
        print(f"Total chunks: {len(smart_chunks)}")
        print(f"Average tokens per chunk: {sum(c['tokens'] for c in smart_chunks) / len(smart_chunks):.1f}")
        print(f"Chunks with bounding boxes: {sum(1 for c in smart_chunks if c['bounding_box'] is not None)}/{len(smart_chunks)}")
        
        # Show chunk distribution by section
        section_counts = defaultdict(int)
        for chunk in smart_chunks:
            section_counts[chunk['section']] += 1
        
        print(f"Chunks per section:")
        for section, count in sorted(section_counts.items()):
            print(f"  {section}: {count} chunks")
        
        # Show sample chunks
        print(f"\nSample chunks:")
        for i, chunk in enumerate(smart_chunks[:5]):
            print(f"  {i+1}. {chunk['id']} ({chunk['tokens']} tokens)")
            print(f"     Text: {chunk['text'][:80]}...")
            print(f"     BBox: page {chunk['pages'][0]}, ({chunk['bounding_box']['left']:.1f}, {chunk['bounding_box']['top']:.1f}, {chunk['bounding_box']['right']:.1f}, {chunk['bounding_box']['bottom']:.1f})")
        
        with open(self.output_dir / "improved_chunks_smart_hybrid.json", 'w', encoding='utf-8') as f:
            json.dump(smart_chunks, f, indent=2)
        print(f"✅ Exported {len(smart_chunks)} chunks to improved_chunks_smart_hybrid.json")
        
        # Adaptive section chunking
        print("\n=== ADAPTIVE SECTION CHUNKING ===")
        adaptive_chunks = self.chunk_adaptive_sections(max_tokens=512)
        print(f"Total chunks: {len(adaptive_chunks)}")
        print(f"Average tokens per chunk: {sum(c['tokens'] for c in adaptive_chunks) / len(adaptive_chunks):.1f}")
        print(f"Chunks with bounding boxes: {sum(1 for c in adaptive_chunks if c['bounding_box'] is not None)}/{len(adaptive_chunks)}")
        
        with open(self.output_dir / "improved_chunks_adaptive_sections.json", 'w', encoding='utf-8') as f:
            json.dump(adaptive_chunks, f, indent=2)
        print(f"✅ Exported {len(adaptive_chunks)} chunks to improved_chunks_adaptive_sections.json")
        
        print("\n🎯 RECOMMENDATIONS:")
        print("1. Use Smart Hybrid for balanced chunking (~30 chunks)")
        print("2. Use Adaptive Sections for section-aware chunking")
        print("3. Both strategies provide accurate positional data")
        print("4. Chunks are semantically coherent and readable")
        print("5. Token limits prevent oversized chunks")


if __name__ == "__main__":
    pdf_file = Path("../../outputs/pdfs/Zj8UqVxClT.pdf")
    output_folder = Path("improved_chunking_output_zj8uqvxclt")
    chunker = ImprovedPositionalChunker(pdf_file, output_folder)
    chunker.run_improved_chunking()
