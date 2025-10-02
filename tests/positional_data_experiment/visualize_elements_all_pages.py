#!/usr/bin/env python3
"""
Visualize Docling elements for all pages of Zj8UqVxClT.pdf
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any
from pathlib import Path
import importlib.util
import os
import sys

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Dynamically import 2_parse_pdf.py
spec = importlib.util.spec_from_file_location("parse_pdf_module", "../../app/2_parse_pdf.py")
parse_pdf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse_pdf_module)

def get_docling_elements(pdf_path: Path, output_dir: Path):
    """Parses PDF with Docling and returns the dictionary output's text elements."""
    docling_dict_output_path = output_dir / "docling_dict_output.json"
    if not docling_dict_output_path.exists():
        print("Converting PDF with Docling to get elements...")
        converter = parse_pdf_module.DocumentConverter(format_options={
            parse_pdf_module.InputFormat.PDF: parse_pdf_module.PdfFormatOption(
                pipeline_options=parse_pdf_module.PdfPipelineOptions()
            )
        })
        result = converter.convert(str(pdf_path))
        doc_dict = result.document.export_to_dict()
        with open(docling_dict_output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, indent=2)
        return doc_dict.get('texts', [])
    else:
        with open(docling_dict_output_path, 'r', encoding='utf-8') as f:
            doc_dict = json.load(f)
            return doc_dict.get('texts', [])

def get_pdf_page_size(pdf_path: Path, output_dir: Path):
    """Gets the page size from Docling's output."""
    docling_dict_output_path = output_dir / "docling_dict_output.json"
    if not docling_dict_output_path.exists():
        # Ensure the PDF is processed first
        get_docling_elements(pdf_path, output_dir)
    
    with open(docling_dict_output_path, 'r', encoding='utf-8') as f:
        doc_dict = json.load(f)
        first_page_info = doc_dict.get('pages', {}).get('1')
        if first_page_info and 'size' in first_page_info:
            size = first_page_info['size']
            if isinstance(size, list) and len(size) >= 2:
                return float(size[0]), float(size[1])
            elif isinstance(size, dict) and 'width' in size and 'height' in size:
                return float(size['width']), float(size['height'])
    return 612.0, 792.0  # Default A4 size if not found

def visualize_elements_on_page(elements: List[Dict], page_number: int, output_filename: str, page_width: float, page_height: float):
    """
    Visualizes bounding boxes of individual Docling elements on a specific page.
    Assumes BOTTOMLEFT coordinate origin for PDF.
    """
    fig, ax = plt.subplots(1, figsize=(page_width / 72, page_height / 72)) # Scale for better visualization
    ax.set_xlim(0, page_width)
    ax.set_ylim(0, page_height)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Docling Elements Visualization - Page {page_number}", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate (PDF points)", fontsize=10)
    ax.set_ylabel("Y Coordinate (PDF points)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    colors = {
        'page_header': 'red',
        'section_header': 'blue',
        'text': 'green',
        'page_footer': 'purple',
        'list_item': 'orange',
        'footnote': 'brown',
        'formula': 'pink',
        'caption': 'gray',
        'code': 'cyan',
        'unspecified': 'black'
    }

    legend_patches = []
    for element_type, color in colors.items():
        legend_patches.append(patches.Patch(color=color, label=element_type))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

    elements_on_current_page = []
    for element in elements:
        if 'prov' in element and element['prov']:
            for prov_item in element['prov']:
                if prov_item['page_no'] == page_number and 'bbox' in prov_item:
                    bbox = prov_item['bbox']
                    rect_x = bbox['l']
                    rect_y = bbox['b'] # Bottom coordinate
                    rect_width = bbox['r'] - bbox['l']
                    rect_height = bbox['t'] - bbox['b'] # Height from bottom to top

                    color = colors.get(element.get('label', 'unspecified'), 'black')
                    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                            linewidth=1, edgecolor=color, facecolor='none', alpha=0.7)
                    ax.add_patch(rect)
                    
                    elements_on_current_page.append({
                        'label': element.get('label', 'unspecified'),
                        'text': element.get('text', '')[:50],
                        'bbox': bbox
                    })
    
    # Sort elements for display from top to bottom
    elements_on_current_page.sort(key=lambda x: x['bbox']['t'], reverse=True)

    print(f"\n🔍 COORDINATE SYSTEM ANALYSIS")
    print("==================================================")
    page_bboxes = [e['bbox'] for el in elements if 'prov' in el and el['prov'] for e in el['prov'] if e['page_no'] == page_number]
    if page_bboxes:
        min_x = min(b['l'] for b in page_bboxes)
        max_x = max(b['r'] for b in page_bboxes)
        min_y = min(b['b'] for b in page_bboxes)
        max_y = max(b['t'] for b in page_bboxes)
        print(f"Page {page_number} bounding boxes: {len(page_bboxes)}")
        print(f"Coordinate ranges:")
        print(f"  X: {min_x:.1f} - {max_x:.1f}")
        print(f"  Y: {min_y:.1f} - {max_y:.1f}")
    
    print(f"\n📊 Found {len(elements_on_current_page)} elements on page {page_number}")

    print(f"\n📋 ELEMENT DETAILS (top to bottom):")
    print("============================================================")
    for i, el in enumerate(elements_on_current_page):
        bbox = el['bbox']
        print(f" {i+1:2d}. {el['label']:15s} | {el['text']:30s} | ({bbox['l']:5.1f}, {bbox['t']:6.1f}, {bbox['r']:5.1f}, {bbox['b']:6.1f})")

    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Created element visualization with {len(elements_on_current_page)} elements")
    print(f"📁 Saved as: {output_filename}")

def main():
    """Main function to visualize elements on all pages."""
    pdf_file = Path("../../outputs/pdfs/Zj8UqVxClT.pdf")
    output_folder = Path("docling_analysis_output_zj8uqvxclt")
    output_folder.mkdir(parents=True, exist_ok=True)

    elements = get_docling_elements(pdf_file, output_folder)
    page_width, page_height = get_pdf_page_size(pdf_file, output_folder)
    
    # Get all pages that have elements
    pages_with_elements = set()
    for element in elements:
        if 'prov' in element and element['prov']:
            for prov_item in element['prov']:
                pages_with_elements.add(prov_item['page_no'])
    
    print(f"📊 DOCLING ELEMENTS VISUALIZATION")
    print(f"==================================================")
    print(f"Total elements: {len(elements)}")
    print(f"Pages with elements: {len(pages_with_elements)}")
    print(f"Page dimensions: {page_width} x {page_height} points")
    
    # Visualize each page
    for page_num in sorted(pages_with_elements):
        output_filename = f"page_{page_num}_elements_layout.png"
        visualize_elements_on_page(elements, page_number=page_num, output_filename=output_filename, page_width=page_width, page_height=page_height)
    
    print(f"\n🎉 SUCCESS!")
    print(f"Visualized elements on {len(pages_with_elements)} pages")
    print(f"Images saved in current directory")

if __name__ == "__main__":
    main()
