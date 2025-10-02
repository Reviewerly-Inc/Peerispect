#!/usr/bin/env python3
"""
Visualize how bounding boxes map to PDF content
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any

def visualize_page_layout(chunks_data: Dict[str, Any], page_number: int = 1):
    """Create a visual representation of bounding boxes on a page"""
    
    # Filter chunks for the specified page
    page_chunks = []
    for chunk in chunks_data['chunks']:
        if page_number in chunk['page_numbers']:
            for bbox in chunk['bounding_boxes']:
                if bbox['page'] == page_number:
                    page_chunks.append({
                        'chunk_id': chunk['id'],
                        'chunk_type': chunk['chunk_type'],
                        'text': chunk['text'][:50] + '...' if len(chunk['text']) > 50 else chunk['text'],
                        'bbox': bbox
                    })
    
    if not page_chunks:
        print(f"No chunks found for page {page_number}")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    
    # Set up coordinate system (BOTTOMLEFT origin)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 800)
    ax.set_aspect('equal')
    
    # Color map for different chunk types
    colors = {
        'section': 'red',
        'element': 'blue',
        'hybrid_element': 'green',
        'hybrid': 'purple'
    }
    
    # Draw bounding boxes
    for i, chunk in enumerate(page_chunks):
        bbox = chunk['bbox']
        
        # Convert coordinates (assuming PDF coordinates)
        x = bbox['left']
        y = bbox['bottom']  # BOTTOMLEFT origin
        width = bbox['right'] - bbox['left']
        height = bbox['top'] - bbox['bottom']
        
        # Create rectangle
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=1,
            edgecolor=colors.get(chunk['chunk_type'], 'black'),
            facecolor=colors.get(chunk['chunk_type'], 'black'),
            alpha=0.3
        )
        ax.add_patch(rect)
        
        # Add text label
        ax.text(x, y + height/2, f"{i+1}: {chunk['chunk_type']}", 
                fontsize=8, ha='left', va='center')
    
    ax.set_title(f'Page {page_number} - Bounding Boxes Visualization')
    ax.set_xlabel('X Coordinate (PDF points)')
    ax.set_ylabel('Y Coordinate (PDF points)')
    
    # Add legend
    legend_elements = [patches.Patch(color=color, label=chunk_type) 
                      for chunk_type, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'page_{page_number}_layout.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualized {len(page_chunks)} chunks on page {page_number}")
    print(f"📊 Chunk types: {set(chunk['chunk_type'] for chunk in page_chunks)}")

def analyze_chunk_distribution(chunks_data: Dict[str, Any]):
    """Analyze how chunks are distributed across pages"""
    
    page_stats = {}
    
    for chunk in chunks_data['chunks']:
        for page_num in chunk['page_numbers']:
            if page_num not in page_stats:
                page_stats[page_num] = {
                    'total_chunks': 0,
                    'chunk_types': {},
                    'total_text_length': 0
                }
            
            stats = page_stats[page_num]
            stats['total_chunks'] += 1
            stats['total_text_length'] += len(chunk['text'])
            
            chunk_type = chunk['chunk_type']
            if chunk_type not in stats['chunk_types']:
                stats['chunk_types'][chunk_type] = 0
            stats['chunk_types'][chunk_type] += 1
    
    print("📊 CHUNK DISTRIBUTION ACROSS PAGES")
    print("=" * 50)
    
    for page_num in sorted(page_stats.keys()):
        stats = page_stats[page_num]
        print(f"Page {page_num}:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Text length: {stats['total_text_length']} chars")
        print(f"  Chunk types: {stats['chunk_types']}")
        print()

def demonstrate_highlighting_usage(chunks_data: Dict[str, Any]):
    """Demonstrate how chunks can be used for frontend highlighting"""
    
    print("🎯 FRONTEND HIGHLIGHTING USAGE EXAMPLES")
    print("=" * 50)
    
    # Show sample chunks with their highlighting data
    sample_chunks = chunks_data['chunks'][:5]
    
    for i, chunk in enumerate(sample_chunks):
        print(f"\nChunk {i+1}: {chunk['id']}")
        print(f"Type: {chunk['chunk_type']}")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Pages: {chunk['page_numbers']}")
        
        if chunk['bounding_boxes']:
            print("Highlighting data:")
            for j, bbox in enumerate(chunk['bounding_boxes']):
                print(f"  Box {j+1}:")
                print(f"    Page: {bbox['page']}")
                print(f"    Coordinates: ({bbox['left']:.1f}, {bbox['top']:.1f}, {bbox['right']:.1f}, {bbox['bottom']:.1f})")
                print(f"    Size: {bbox['right'] - bbox['left']:.1f} x {bbox['top'] - bbox['bottom']:.1f}")
                if 'char_span' in bbox:
                    print(f"    Character span: {bbox['char_span']}")
        
        print()

def main():
    """Main demonstration function"""
    print("🎨 Visualizing positional data for frontend highlighting...")
    
    # Load chunk data
    try:
        with open('chunks_hybrid.json', 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        print(f"✅ Loaded {chunks_data['metadata']['total_chunks']} chunks")
    except FileNotFoundError:
        print("❌ No chunk data found. Run chunking_with_positional_data.py first.")
        return
    
    # Analyze distribution
    analyze_chunk_distribution(chunks_data)
    
    # Demonstrate highlighting usage
    demonstrate_highlighting_usage(chunks_data)
    
    # Visualize page layout (if matplotlib is available)
    try:
        visualize_page_layout(chunks_data, page_number=1)
    except ImportError:
        print("📊 Matplotlib not available for visualization")
        print("   Install with: pip install matplotlib")
    
    print("\n🎯 KEY INSIGHTS:")
    print("1. Bounding boxes provide exact PDF coordinates for highlighting")
    print("2. Character spans enable precise text selection")
    print("3. Page numbers allow multi-page document handling")
    print("4. Chunk types provide semantic information for styling")
    print("5. Coordinates are in PDF points (1/72 inch)")

if __name__ == "__main__":
    main()
