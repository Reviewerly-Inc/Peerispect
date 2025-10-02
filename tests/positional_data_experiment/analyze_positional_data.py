#!/usr/bin/env python3
"""
Analyze Docling's positional data and its relationship to markdown structure
"""

import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates"""
    left: float
    top: float
    right: float
    bottom: float
    page: int
    char_span: Tuple[int, int]
    
    @property
    def width(self) -> float:
        return self.right - self.left
    
    @property
    def height(self) -> float:
        return self.bottom - self.top
    
    @property
    def area(self) -> float:
        return self.width * self.height

@dataclass
class TextElement:
    """Represents a text element with positional data"""
    text: str
    label: str
    bounding_boxes: List[BoundingBox]
    content_layer: str

class PositionalDataAnalyzer:
    def __init__(self, docling_json_path: str, markdown_path: str):
        """Initialize analyzer with Docling JSON and markdown files"""
        with open(docling_json_path, 'r', encoding='utf-8') as f:
            self.docling_data = json.load(f)
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            self.markdown_content = f.read()
        
        self.text_elements = self._parse_text_elements()
        self.markdown_structure = self._parse_markdown_structure()
    
    def _parse_text_elements(self) -> List[TextElement]:
        """Parse text elements from Docling data"""
        elements = []
        
        for text_data in self.docling_data['texts']:
            bounding_boxes = []
            
            for prov in text_data.get('prov', []):
                bbox_data = prov['bbox']
                bbox = BoundingBox(
                    left=bbox_data['l'],
                    top=bbox_data['t'],
                    right=bbox_data['r'],
                    bottom=bbox_data['b'],
                    page=prov['page_no'],
                    char_span=tuple(prov['charspan'])
                )
                bounding_boxes.append(bbox)
            
            element = TextElement(
                text=text_data['text'],
                label=text_data.get('label', 'unknown'),
                bounding_boxes=bounding_boxes,
                content_layer=text_data.get('content_layer', 'unknown')
            )
            elements.append(element)
        
        return elements
    
    def _parse_markdown_structure(self) -> Dict[str, Any]:
        """Parse markdown structure to identify sections"""
        lines = self.markdown_content.split('\n')
        
        structure = {
            'headers': [],
            'sections': [],
            'paragraphs': [],
            'lists': []
        }
        
        current_section = None
        current_paragraph = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    structure['sections'].append({
                        'header': current_section,
                        'content_lines': current_paragraph,
                        'start_line': current_section['line_number']
                    })
                
                # Start new section
                level = len(line) - len(line.lstrip('#'))
                current_section = {
                    'text': line,
                    'level': level,
                    'line_number': i
                }
                structure['headers'].append(current_section)
                current_paragraph = []
            
            elif line and current_section:
                current_paragraph.append({
                    'text': line,
                    'line_number': i
                })
        
        # Add final section
        if current_section:
            structure['sections'].append({
                'header': current_section,
                'content_lines': current_paragraph,
                'start_line': current_section['line_number']
            })
        
        return structure
    
    def analyze_coordinate_system(self) -> Dict[str, Any]:
        """Analyze the coordinate system and ranges"""
        all_boxes = []
        for element in self.text_elements:
            all_boxes.extend(element.bounding_boxes)
        
        if not all_boxes:
            return {}
        
        return {
            'total_boxes': len(all_boxes),
            'pages': list(set(box.page for box in all_boxes)),
            'coordinate_ranges': {
                'left': (min(box.left for box in all_boxes), max(box.left for box in all_boxes)),
                'top': (min(box.top for box in all_boxes), max(box.top for box in all_boxes)),
                'right': (min(box.right for box in all_boxes), max(box.right for box in all_boxes)),
                'bottom': (min(box.bottom for box in all_boxes), max(box.bottom for box in all_boxes))
            },
            'box_sizes': {
                'min_width': min(box.width for box in all_boxes),
                'max_width': max(box.width for box in all_boxes),
                'min_height': min(box.height for box in all_boxes),
                'max_height': max(box.height for box in all_boxes),
                'avg_width': sum(box.width for box in all_boxes) / len(all_boxes),
                'avg_height': sum(box.height for box in all_boxes) / len(all_boxes)
            }
        }
    
    def analyze_element_types(self) -> Dict[str, Any]:
        """Analyze different element types and their characteristics"""
        type_stats = {}
        
        for element in self.text_elements:
            label = element.label
            if label not in type_stats:
                type_stats[label] = {
                    'count': 0,
                    'total_text_length': 0,
                    'avg_text_length': 0,
                    'bounding_boxes': [],
                    'pages': set()
                }
            
            stats = type_stats[label]
            stats['count'] += 1
            stats['total_text_length'] += len(element.text)
            stats['bounding_boxes'].extend(element.bounding_boxes)
            stats['pages'].update(box.page for box in element.bounding_boxes)
        
        # Calculate averages
        for label, stats in type_stats.items():
            stats['avg_text_length'] = stats['total_text_length'] / stats['count']
            stats['pages'] = sorted(list(stats['pages']))
            stats['unique_pages'] = len(stats['pages'])
        
        return type_stats
    
    def find_section_bounding_boxes(self) -> List[Dict[str, Any]]:
        """Find bounding boxes that correspond to markdown sections"""
        section_boxes = []
        
        for section in self.markdown_structure['sections']:
            header_text = section['header']['text'].replace('#', '').strip()
            
            # Find matching section header elements
            matching_elements = []
            for element in self.text_elements:
                if element.label == 'section_header':
                    # Check if text matches (allowing for some variation)
                    element_text = element.text.strip()
                    if (header_text.lower() in element_text.lower() or 
                        element_text.lower() in header_text.lower()):
                        matching_elements.append(element)
            
            if matching_elements:
                # Get all bounding boxes for this section
                section_bboxes = []
                for element in matching_elements:
                    section_bboxes.extend(element.bounding_boxes)
                
                section_boxes.append({
                    'header': header_text,
                    'markdown_line': section['header']['line_number'],
                    'elements': len(matching_elements),
                    'bounding_boxes': section_bboxes,
                    'page_range': (min(box.page for box in section_bboxes), 
                                 max(box.page for box in section_bboxes)) if section_bboxes else (0, 0)
                })
        
        return section_boxes
    
    def analyze_chunking_opportunities(self) -> Dict[str, Any]:
        """Analyze how bounding boxes could be used for chunking"""
        opportunities = {
            'section_level': [],
            'paragraph_level': [],
            'element_level': [],
            'character_level': []
        }
        
        # Section-level chunking
        section_boxes = self.find_section_bounding_boxes()
        for section in section_boxes:
            if section['bounding_boxes']:
                # Calculate section bounding box
                all_boxes = section['bounding_boxes']
                section_bbox = BoundingBox(
                    left=min(box.left for box in all_boxes),
                    top=min(box.top for box in all_boxes),
                    right=max(box.right for box in all_boxes),
                    bottom=max(box.bottom for box in all_boxes),
                    page=min(box.page for box in all_boxes),
                    char_span=(0, 0)  # Not applicable for section-level
                )
                
                opportunities['section_level'].append({
                    'header': section['header'],
                    'bounding_box': section_bbox,
                    'element_count': len(all_boxes)
                })
        
        # Element-level chunking
        for element in self.text_elements:
            if element.bounding_boxes:
                opportunities['element_level'].append({
                    'text': element.text[:100] + '...' if len(element.text) > 100 else element.text,
                    'label': element.label,
                    'bounding_boxes': element.bounding_boxes,
                    'text_length': len(element.text)
                })
        
        # Character-level chunking (using char_span)
        char_level_chunks = []
        for element in self.text_elements:
            for bbox in element.bounding_boxes:
                if bbox.char_span[1] - bbox.char_span[0] > 0:
                    char_level_chunks.append({
                        'text': element.text[bbox.char_span[0]:bbox.char_span[1]],
                        'char_span': bbox.char_span,
                        'bounding_box': bbox,
                        'label': element.label
                    })
        
        opportunities['character_level'] = char_level_chunks[:10]  # Sample
        
        return opportunities
    
    def generate_chunking_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for integrating positional data with chunking"""
        coordinate_analysis = self.analyze_coordinate_system()
        element_analysis = self.analyze_element_types()
        chunking_opportunities = self.analyze_chunking_opportunities()
        
        recommendations = {
            'coordinate_system': {
                'origin': 'BOTTOMLEFT',
                'page_height': coordinate_analysis['coordinate_ranges']['top'][1] - coordinate_analysis['coordinate_ranges']['bottom'][0],
                'page_width': coordinate_analysis['coordinate_ranges']['right'][1] - coordinate_analysis['coordinate_ranges']['left'][0],
                'note': 'Coordinates are in PDF points, origin at bottom-left'
            },
            'chunking_strategies': {
                'section_level': {
                    'feasible': len(chunking_opportunities['section_level']) > 0,
                    'count': len(chunking_opportunities['section_level']),
                    'description': 'Group elements by markdown sections',
                    'pros': ['Semantic coherence', 'Easy to implement'],
                    'cons': ['May create large chunks', 'Sections vary in size']
                },
                'element_level': {
                    'feasible': len(chunking_opportunities['element_level']) > 0,
                    'count': len(chunking_opportunities['element_level']),
                    'description': 'Each Docling element becomes a chunk',
                    'pros': ['Granular control', 'Preserves element types'],
                    'cons': ['May create too many small chunks', 'Loss of context']
                },
                'character_level': {
                    'feasible': len(chunking_opportunities['character_level']) > 0,
                    'count': len(chunking_opportunities['character_level']),
                    'description': 'Use char_span for precise text mapping',
                    'pros': ['Most precise', 'Character-level highlighting'],
                    'cons': ['Complex implementation', 'May fragment meaning']
                }
            },
            'implementation_notes': [
                'Bounding boxes are in PDF coordinate system (BOTTOMLEFT origin)',
                'Each text element can span multiple pages',
                'Character spans provide precise text mapping within elements',
                'Element labels provide semantic information for chunking decisions',
                'Consider combining strategies: section-level for structure, element-level for content'
            ]
        }
        
        return recommendations

def main():
    """Main analysis function"""
    print("🔍 Analyzing Docling's positional data structure...")
    
    # Initialize analyzer
    analyzer = PositionalDataAnalyzer(
        '../mineru_experiment/outputs/docling_dict_output.json',
        '../mineru_experiment/outputs/docling_comparison.md'
    )
    
    print(f"📊 Loaded {len(analyzer.text_elements)} text elements")
    print(f"📄 Markdown has {len(analyzer.markdown_structure['headers'])} headers")
    
    # Analyze coordinate system
    print("\n=== COORDINATE SYSTEM ANALYSIS ===")
    coord_analysis = analyzer.analyze_coordinate_system()
    print(f"Total bounding boxes: {coord_analysis['total_boxes']}")
    print(f"Pages: {coord_analysis['pages']}")
    print(f"Coordinate ranges:")
    for coord, (min_val, max_val) in coord_analysis['coordinate_ranges'].items():
        print(f"  {coord}: {min_val:.1f} - {max_val:.1f}")
    
    # Analyze element types
    print("\n=== ELEMENT TYPE ANALYSIS ===")
    element_analysis = analyzer.analyze_element_types()
    for label, stats in element_analysis.items():
        print(f"{label}: {stats['count']} elements, avg length: {stats['avg_text_length']:.1f} chars")
    
    # Find section bounding boxes
    print("\n=== SECTION BOUNDING BOXES ===")
    section_boxes = analyzer.find_section_bounding_boxes()
    for section in section_boxes:
        print(f"Section: {section['header']}")
        print(f"  Elements: {section['elements']}, Pages: {section['page_range']}")
        if section['bounding_boxes']:
            bbox = section['bounding_boxes'][0]
            print(f"  Sample bbox: ({bbox.left:.1f}, {bbox.top:.1f}, {bbox.right:.1f}, {bbox.bottom:.1f})")
    
    # Generate recommendations
    print("\n=== CHUNKING RECOMMENDATIONS ===")
    recommendations = analyzer.generate_chunking_recommendations()
    
    print("Coordinate System:")
    print(f"  Origin: {recommendations['coordinate_system']['origin']}")
    print(f"  Page size: {recommendations['coordinate_system']['page_width']:.1f} x {recommendations['coordinate_system']['page_height']:.1f}")
    
    print("\nChunking Strategies:")
    for strategy, info in recommendations['chunking_strategies'].items():
        status = "✅" if info['feasible'] else "❌"
        print(f"  {status} {strategy}: {info['description']}")
        print(f"    Count: {info['count']}, Feasible: {info['feasible']}")
    
    print("\nImplementation Notes:")
    for note in recommendations['implementation_notes']:
        print(f"  • {note}")

if __name__ == "__main__":
    main()
