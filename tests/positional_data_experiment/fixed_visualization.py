#!/usr/bin/env python3
"""
Fixed visualization that properly maps Docling bounding boxes to PDF layout
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any

def create_corrected_visualization():
    """Create a corrected visualization that properly maps to PDF layout"""
    
    # Load original Docling data
    with open('../mineru_experiment/outputs/docling_dict_output.json', 'r', encoding='utf-8') as f:
        docling_data = json.load(f)
    
    # Filter elements for page 1 only
    page1_elements = []
    for text in docling_data['texts']:
        for prov in text.get('prov', []):
            if prov['page_no'] == 1:
                bbox = prov['bbox']
                element = {
                    'text': text['text'][:30] + '...' if len(text['text']) > 30 else text['text'],
                    'label': text['label'],
                    'left': bbox['l'],
                    'top': bbox['t'],
                    'right': bbox['r'],
                    'bottom': bbox['b'],
                    'char_span': prov['charspan']
                }
                page1_elements.append(element)
    
    # Sort elements by Y coordinate (top to bottom for BOTTOMLEFT origin)
    page1_elements.sort(key=lambda x: x['top'], reverse=True)
    
    print(f"📊 Found {len(page1_elements)} elements on page 1")
    
    # Create figure with correct aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    
    # Set up coordinate system based on actual PDF dimensions
    # From analysis: width ~400, height ~733
    ax.set_xlim(0, 520)  # Slightly larger than max right coordinate
    ax.set_ylim(0, 800)  # Slightly larger than max top coordinate
    ax.set_aspect('equal')
    
    # Color map for different element types
    colors = {
        'page_header': 'lightblue',
        'section_header': 'red',
        'text': 'lightgreen',
        'page_footer': 'orange',
        'list_item': 'yellow',
        'footnote': 'purple',
        'formula': 'pink',
        'caption': 'cyan',
        'code': 'brown'
    }
    
    # Draw bounding boxes
    for i, element in enumerate(page1_elements):
        # Convert BOTTOMLEFT coordinates to matplotlib coordinates
        x = element['left']
        y = element['bottom']  # BOTTOMLEFT origin
        width = element['right'] - element['left']
        height = element['top'] - element['bottom']
        
        # Create rectangle
        color = colors.get(element['label'], 'gray')
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.6
        )
        ax.add_patch(rect)
        
        # Add text label
        ax.text(x + 2, y + height/2, f"{i+1}: {element['label']}", 
                fontsize=8, ha='left', va='center', weight='bold')
        
        # Add text preview
        ax.text(x + 2, y + height/2 - 8, element['text'], 
                fontsize=6, ha='left', va='center')
    
    # Add grid for better visualization
    ax.grid(True, alpha=0.3)
    
    ax.set_title('Page 1 - Corrected Bounding Boxes Visualization\\n(BOTTOMLEFT Origin)', 
                 fontsize=14, weight='bold')
    ax.set_xlabel('X Coordinate (PDF points)', fontsize=12)
    ax.set_ylabel('Y Coordinate (PDF points)', fontsize=12)
    
    # Add legend
    legend_elements = [patches.Patch(color=color, label=label) 
                      for label, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add coordinate info
    ax.text(0.02, 0.98, f'Elements: {len(page1_elements)}\\nOrigin: BOTTOMLEFT\\nPage size: ~400 x 733 pts', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('page_1_corrected_layout.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print element details
    print("\\n📋 ELEMENT DETAILS (top to bottom):")
    print("=" * 60)
    for i, element in enumerate(page1_elements):
        print(f"{i+1:2d}. {element['label']:15s} | {element['text']:30s} | "
              f"({element['left']:6.1f}, {element['top']:6.1f}, {element['right']:6.1f}, {element['bottom']:6.1f})")
    
    return page1_elements

def analyze_coordinate_issues():
    """Analyze why the original visualization was wrong"""
    
    print("\\n🔍 COORDINATE SYSTEM ANALYSIS")
    print("=" * 50)
    
    # Load original Docling data
    with open('../mineru_experiment/outputs/docling_dict_output.json', 'r', encoding='utf-8') as f:
        docling_data = json.load(f)
    
    # Get all bounding boxes for page 1
    page1_boxes = []
    for text in docling_data['texts']:
        for prov in text.get('prov', []):
            if prov['page_no'] == 1:
                bbox = prov['bbox']
                page1_boxes.append({
                    'label': text['label'],
                    'left': bbox['l'],
                    'top': bbox['t'],
                    'right': bbox['r'],
                    'bottom': bbox['b']
                })
    
    if page1_boxes:
        print(f"Page 1 bounding boxes: {len(page1_boxes)}")
        print(f"Coordinate ranges:")
        print(f"  X: {min(b['left'] for b in page1_boxes):.1f} - {max(b['right'] for b in page1_boxes):.1f}")
        print(f"  Y: {min(b['bottom'] for b in page1_boxes):.1f} - {max(b['top'] for b in page1_boxes):.1f}")
        
        print(f"\\nIssues with original visualization:")
        print(f"1. ❌ Wrong coordinate system - used TOPLEFT instead of BOTTOMLEFT")
        print(f"2. ❌ Wrong scaling - didn't account for actual PDF dimensions")
        print(f"3. ❌ Duplicate chunks - chunking logic created duplicates")
        print(f"4. ❌ Wrong element grouping - mixed different element types")
        
        print(f"\\nCorrected approach:")
        print(f"1. ✅ Use BOTTOMLEFT origin (y = bottom, height = top - bottom)")
        print(f"2. ✅ Use actual PDF dimensions (~400 x 733 points)")
        print(f"3. ✅ Show individual elements, not grouped chunks")
        print(f"4. ✅ Sort by Y coordinate for proper top-to-bottom layout")

def main():
    """Main function to create corrected visualization"""
    print("🔧 Creating corrected bounding box visualization...")
    
    # Analyze the issues
    analyze_coordinate_issues()
    
    # Create corrected visualization
    elements = create_corrected_visualization()
    
    print(f"\\n✅ Created corrected visualization with {len(elements)} elements")
    print("📁 Saved as: page_1_corrected_layout.png")

if __name__ == "__main__":
    main()
