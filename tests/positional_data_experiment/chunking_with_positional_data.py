#!/usr/bin/env python3
"""
Example implementation of chunking with positional data integration
"""

import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

@dataclass
class PositionalChunk:
    """A chunk with positional data for frontend highlighting"""
    id: str
    text: str
    chunk_type: str  # 'section', 'paragraph', 'element', 'custom'
    bounding_boxes: List[Dict[str, Any]]  # List of bbox dicts
    page_numbers: List[int]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

class PositionalChunker:
    def __init__(self, docling_json_path: str, markdown_path: str):
        """Initialize chunker with Docling data"""
        with open(docling_json_path, 'r', encoding='utf-8') as f:
            self.docling_data = json.load(f)
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            self.markdown_content = f.read()
        
        self.text_elements = self._parse_text_elements()
    
    def _parse_text_elements(self) -> List[Dict[str, Any]]:
        """Parse text elements from Docling data"""
        elements = []
        for text_data in self.docling_data['texts']:
            element = {
                'text': text_data['text'],
                'label': text_data.get('label', 'unknown'),
                'content_layer': text_data.get('content_layer', 'unknown'),
                'bounding_boxes': []
            }
            
            for prov in text_data.get('prov', []):
                bbox = {
                    'page': prov['page_no'],
                    'left': prov['bbox']['l'],
                    'top': prov['bbox']['t'],
                    'right': prov['bbox']['r'],
                    'bottom': prov['bbox']['b'],
                    'char_span': prov['charspan']
                }
                element['bounding_boxes'].append(bbox)
            
            elements.append(element)
        
        return elements
    
    def chunk_by_sections(self, max_chunk_size: int = 1000) -> List[PositionalChunk]:
        """Create chunks based on markdown sections"""
        chunks = []
        markdown_lines = self.markdown_content.split('\n')
        
        current_section = None
        current_text = []
        current_elements = []
        chunk_id = 0
        
        for i, line in enumerate(markdown_lines):
            line = line.strip()
            
            if line.startswith('#'):
                # Save previous section
                if current_section and current_text:
                    chunk = self._create_section_chunk(
                        chunk_id, current_section, current_text, current_elements
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new section
                current_section = line
                current_text = [line]
                current_elements = self._find_elements_for_section(line)
            
            elif line and current_section:
                current_text.append(line)
                
                # Check if we need to split due to size
                if len('\n'.join(current_text)) > max_chunk_size:
                    # Create chunk for current content
                    chunk = self._create_section_chunk(
                        chunk_id, current_section, current_text, current_elements
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    
                    # Start new sub-chunk
                    current_text = [line]
        
        # Add final section
        if current_section and current_text:
            chunk = self._create_section_chunk(
                chunk_id, current_section, current_text, current_elements
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_elements(self, group_similar: bool = True) -> List[PositionalChunk]:
        """Create chunks based on individual elements"""
        chunks = []
        
        if group_similar:
            # Group similar elements together
            grouped_elements = self._group_similar_elements()
            for group_type, elements in grouped_elements.items():
                for i, element in enumerate(elements):
                    chunk = PositionalChunk(
                        id=f"element_{group_type}_{i}",
                        text=element['text'],
                        chunk_type='element',
                        bounding_boxes=element['bounding_boxes'],
                        page_numbers=list(set(bbox['page'] for bbox in element['bounding_boxes'])),
                        metadata={
                            'element_label': element['label'],
                            'content_layer': element['content_layer'],
                            'text_length': len(element['text']),
                            'group_type': group_type
                        }
                    )
                    chunks.append(chunk)
        else:
            # One chunk per element
            for i, element in enumerate(self.text_elements):
                chunk = PositionalChunk(
                    id=f"element_{i}",
                    text=element['text'],
                    chunk_type='element',
                    bounding_boxes=element['bounding_boxes'],
                    page_numbers=list(set(bbox['page'] for bbox in element['bounding_boxes'])),
                    metadata={
                        'element_label': element['label'],
                        'content_layer': element['content_layer'],
                        'text_length': len(element['text'])
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def chunk_by_hybrid(self, section_max_size: int = 2000) -> List[PositionalChunk]:
        """Hybrid approach: sections for structure, elements for content"""
        chunks = []
        
        # First, get section chunks
        section_chunks = self.chunk_by_sections(section_max_size)
        
        for section_chunk in section_chunks:
            # If section is too large, break it down by elements
            if len(section_chunk.text) > section_max_size:
                # Find elements that belong to this section
                section_elements = self._find_elements_for_section(section_chunk.metadata['section_header'])
                
                # Create element-based sub-chunks
                for i, element in enumerate(section_elements):
                    chunk = PositionalChunk(
                        id=f"hybrid_{section_chunk.id}_{i}",
                        text=element['text'],
                        chunk_type='hybrid_element',
                        bounding_boxes=element['bounding_boxes'],
                        page_numbers=list(set(bbox['page'] for bbox in element['bounding_boxes'])),
                        metadata={
                            'parent_section': section_chunk.metadata['section_header'],
                            'element_label': element['label'],
                            'content_layer': element['content_layer'],
                            'text_length': len(element['text'])
                        }
                    )
                    chunks.append(chunk)
            else:
                chunks.append(section_chunk)
        
        return chunks
    
    def _create_section_chunk(self, chunk_id: int, section_header: str, 
                            text_lines: List[str], elements: List[Dict]) -> PositionalChunk:
        """Create a chunk for a markdown section"""
        text = '\n'.join(text_lines)
        
        # Collect all bounding boxes from elements in this section
        all_bboxes = []
        for element in elements:
            all_bboxes.extend(element['bounding_boxes'])
        
        # Calculate section bounding box
        if all_bboxes:
            section_bbox = {
                'page': min(bbox['page'] for bbox in all_bboxes),
                'left': min(bbox['left'] for bbox in all_bboxes),
                'top': min(bbox['top'] for bbox in all_bboxes),
                'right': max(bbox['right'] for bbox in all_bboxes),
                'bottom': max(bbox['bottom'] for bbox in all_bboxes),
                'char_span': [0, len(text)]  # Full text span
            }
        else:
            section_bbox = {}
        
        return PositionalChunk(
            id=f"section_{chunk_id}",
            text=text,
            chunk_type='section',
            bounding_boxes=[section_bbox] if section_bbox else [],
            page_numbers=list(set(bbox['page'] for bbox in all_bboxes)) if all_bboxes else [],
            metadata={
                'section_header': section_header,
                'element_count': len(elements),
                'text_length': len(text),
                'line_count': len(text_lines)
            }
        )
    
    def _find_elements_for_section(self, section_header: str) -> List[Dict]:
        """Find elements that belong to a section"""
        # Simple matching - could be improved with more sophisticated logic
        header_text = section_header.replace('#', '').strip().lower()
        matching_elements = []
        
        for element in self.text_elements:
            if element['label'] == 'section_header':
                element_text = element['text'].strip().lower()
                if (header_text in element_text or element_text in header_text):
                    # Find all elements that come after this header
                    # This is a simplified approach - in practice, you'd need more sophisticated logic
                    matching_elements.append(element)
        
        return matching_elements
    
    def _group_similar_elements(self) -> Dict[str, List[Dict]]:
        """Group elements by type for more efficient chunking"""
        groups = {}
        
        for element in self.text_elements:
            label = element['label']
            if label not in groups:
                groups[label] = []
            groups[label].append(element)
        
        return groups
    
    def export_chunks(self, chunks: List[PositionalChunk], output_path: str):
        """Export chunks to JSON file"""
        chunks_data = {
            'metadata': {
                'total_chunks': len(chunks),
                'chunk_types': list(set(chunk.chunk_type for chunk in chunks)),
                'total_pages': list(set(page for chunk in chunks for page in chunk.page_numbers))
            },
            'chunks': [chunk.to_dict() for chunk in chunks]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(chunks)} chunks to {output_path}")

def main():
    """Demonstrate different chunking strategies"""
    print("🔧 Demonstrating chunking strategies with positional data...")
    
    # Initialize chunker
    chunker = PositionalChunker(
        '../mineru_experiment/outputs/docling_dict_output.json',
        '../mineru_experiment/outputs/docling_comparison.md'
    )
    
    print(f"📊 Loaded {len(chunker.text_elements)} text elements")
    
    # Test different chunking strategies
    strategies = {
        'section_based': chunker.chunk_by_sections(max_chunk_size=1000),
        'element_based': chunker.chunk_by_elements(group_similar=True),
        'hybrid': chunker.chunk_by_hybrid(section_max_size=2000)
    }
    
    for strategy_name, chunks in strategies.items():
        print(f"\n=== {strategy_name.upper()} CHUNKING ===")
        print(f"Total chunks: {len(chunks)}")
        
        # Analyze chunk characteristics
        chunk_types = {}
        total_text_length = 0
        chunks_with_bboxes = 0
        
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            if chunk_type not in chunk_types:
                chunk_types[chunk_type] = 0
            chunk_types[chunk_type] += 1
            
            total_text_length += len(chunk.text)
            if chunk.bounding_boxes:
                chunks_with_bboxes += 1
        
        print(f"Chunk types: {chunk_types}")
        print(f"Average chunk size: {total_text_length / len(chunks):.1f} characters")
        print(f"Chunks with bounding boxes: {chunks_with_bboxes}/{len(chunks)}")
        
        # Show sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  {i+1}. {chunk.chunk_type} - {chunk.text[:50]}...")
            if chunk.bounding_boxes:
                bbox = chunk.bounding_boxes[0]
                print(f"     BBox: page {bbox['page']}, ({bbox['left']:.1f}, {bbox['top']:.1f}, {bbox['right']:.1f}, {bbox['bottom']:.1f})")
        
        # Export chunks
        output_file = f"chunks_{strategy_name}.json"
        chunker.export_chunks(chunks, output_file)
    
    print("\n🎯 RECOMMENDATIONS:")
    print("1. Use section-based chunking for document structure")
    print("2. Use element-based chunking for granular content")
    print("3. Use hybrid approach for balanced chunking")
    print("4. Bounding boxes enable precise frontend highlighting")
    print("5. Character spans provide exact text mapping")

if __name__ == "__main__":
    main()
