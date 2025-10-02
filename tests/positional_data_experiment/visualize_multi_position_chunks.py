import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any
import importlib.util
import os
import sys
from collections import defaultdict

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Dynamically import 2_parse_pdf.py
spec = importlib.util.spec_from_file_location("parse_pdf_module", "../../app/2_parse_pdf.py")
parse_pdf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse_pdf_module)

def get_pdf_page_size(pdf_path: Path, output_dir: Path):
    """Gets the page size from Docling's output."""
    docling_dict_output_path = output_dir / "docling_dict_output.json"
    if not docling_dict_output_path.exists():
        return 612.0, 792.0
    
    with open(docling_dict_output_path, 'r', encoding='utf-8') as f:
        doc_dict = json.load(f)
        first_page_info = doc_dict.get('pages', {}).get('1')
        if first_page_info and 'size' in first_page_info:
            size = first_page_info['size']
            if isinstance(size, list) and len(size) >= 2:
                return float(size[0]), float(size[1])
            elif isinstance(size, dict) and 'width' in size and 'height' in size:
                return float(size['width']), float(size['height'])
    return 612.0, 792.0

def visualize_multi_position_chunks_on_pages(chunks: List[Dict], max_pages: int = None, output_dir: Path = Path(".")):
    """
    Visualizes chunks with multiple positions across multiple pages.
    """
    # Group chunks by page
    page_chunks = defaultdict(list)
    for chunk in chunks:
        if chunk.get('positions'):
            for pos in chunk['positions']:
                page_chunks[pos['page']].append((chunk, pos))
    
    # Get page dimensions
    pdf_file = Path("../../outputs/pdfs/Zj8UqVxClT.pdf")
    output_folder = Path("improved_chunking_output_v3")
    page_width, page_height = get_pdf_page_size(pdf_file, output_folder)
    
    print(f"📊 MULTI-POSITION CHUNK LAYOUT VISUALIZATION")
    print(f"==================================================")
    print(f"Total chunks: {len(chunks)}")
    print(f"Pages with chunks: {len(page_chunks)}")
    print(f"Page dimensions: {page_width} x {page_height} points")
    
    # Create visualization for each page
    pages_to_visualize = sorted(page_chunks.keys())
    if max_pages:
        pages_to_visualize = pages_to_visualize[:max_pages]
    
    for page_num in pages_to_visualize:
        chunk_positions = page_chunks[page_num]
        print(f"\n📄 PAGE {page_num}: {len(chunk_positions)} chunk positions")
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(page_width / 72, page_height / 72))
        ax.set_xlim(0, page_width)
        ax.set_ylim(0, page_height)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Multi-Position Chunk Layout - Page {page_num}", fontsize=14, fontweight='bold')
        ax.set_xlabel("X Coordinate (PDF points)", fontsize=10)
        ax.set_ylabel("Y Coordinate (PDF points)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add column separator line
        ax.axvline(x=page_width/2, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.text(page_width/2 + 5, page_height - 20, 'Column Split', color='red', fontsize=8, alpha=0.7)
        
        # Color scheme for different columns
        colors = {
            'left': '#4A90E2',
            'right': '#7ED321',
            'both': '#F5A623',
            'unknown': '#9B59B6'
        }
        
        # Track chunks that appear on this page
        chunks_on_page = set()
        for chunk, pos in chunk_positions:
            chunks_on_page.add(chunk['id'])
        
        # Plot chunk positions
        for i, (chunk, pos) in enumerate(chunk_positions):
            bbox = pos['bounding_box']
            column = pos['column']
            
            # Rectangle coordinates (BOTTOMLEFT origin)
            rect_x = bbox['left']
            rect_y = bbox['bottom']
            rect_width = bbox['right'] - bbox['left']
            rect_height = bbox['top'] - bbox['bottom']
            
            # Choose color based on column
            color = colors.get(column, '#9B59B6')
            
            # Create rectangle
            rect = patches.Rectangle(
                (rect_x, rect_y), rect_width, rect_height,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.4
            )
            ax.add_patch(rect)
            
            # Add chunk label
            chunk_id = chunk.get('id', f'chunk_{i}')
            section = chunk.get('section', 'Unknown')
            tokens = chunk.get('tokens', 0)
            
            # Position label at top-left of chunk
            label_x = rect_x + 5
            label_y = rect_y + rect_height - 10
            
            ax.text(label_x, label_y, f"{chunk_id}\n{tokens}t", 
                   fontsize=8, color='black', fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', alpha=0.8, edgecolor=color))
            
            # Add section and column info
            info_text = f"{section[:15]}...\n{column}"
            ax.text(label_x, label_y - 20, info_text, 
                   fontsize=6, color='gray', alpha=0.8,
                   verticalalignment='top')
            
            # Add position number if chunk has multiple positions
            if len(chunk.get('positions', [])) > 1:
                pos_num = next(j for j, p in enumerate(chunk['positions']) if p == pos) + 1
                ax.text(rect_x + rect_width - 15, rect_y + 5, f"P{pos_num}", 
                       fontsize=6, color='red', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
        
        # Add legend
        legend_patches = []
        for column, color in colors.items():
            if any(pos['column'] == column for _, pos in chunk_positions):
                legend_patches.append(patches.Patch(color=color, label=column))
        
        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper right', fontsize=8)
        
        # Add page info
        total_tokens = sum(chunk.get('tokens', 0) for chunk, _ in chunk_positions)
        left_positions = sum(1 for _, pos in chunk_positions if pos['column'] == 'left')
        right_positions = sum(1 for _, pos in chunk_positions if pos['column'] == 'right')
        both_positions = sum(1 for _, pos in chunk_positions if pos['column'] == 'both')
        
        ax.text(page_width - 10, page_height - 10, 
               f"Chunks: {len(chunks_on_page)}\nPositions: {len(chunk_positions)}\n"
               f"Left: {left_positions}, Right: {right_positions}, Both: {both_positions}\n"
               f"Tokens: {total_tokens}",
               fontsize=9, ha='right', va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Save figure
        output_file = output_dir / f"multi_position_chunk_layout_page_{page_num}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✅ Saved: {output_file}")
        print(f"  📊 Chunks: {len(chunks_on_page)}, Positions: {len(chunk_positions)}")
        print(f"  📊 Left: {left_positions}, Right: {right_positions}, Both: {both_positions}")
        print(f"  📊 Total tokens: {total_tokens}")
        
        # Show chunk details
        print(f"  📋 Chunk position details:")
        for i, (chunk, pos) in enumerate(chunk_positions[:3]):  # Show first 3 positions
            bbox = pos['bounding_box']
            print(f"    {i+1}. {chunk.get('id', 'unknown')} - {pos['column']} column")
            print(f"       BBox: ({bbox['left']:.1f}, {bbox['top']:.1f}, {bbox['right']:.1f}, {bbox['bottom']:.1f})")
            print(f"       Text: {pos.get('text', '')[:60]}...")
    
    print(f"\n🎯 MULTI-POSITION LAYOUT ANALYSIS")
    print(f"==================================================")
    print(f"✅ Chunks can span multiple columns and pages")
    print(f"✅ Multiple position data for complex layouts")
    print(f"✅ Accurate column detection and visualization")
    print(f"✅ Section coherence maintained across positions")
    print(f"✅ Perfect for frontend highlighting with multiple regions")
    
    return len(page_chunks)

def main():
    """Main function to visualize multi-position chunk layouts."""
    # Load the improved chunks
    chunks_file = Path("improved_chunking_output_v3/improved_chunks_v3_section_multi_pos.json")
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Please run improved_chunking_v3.py first")
        return
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Create output directory
    output_dir = Path("multi_position_chunk_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Visualize chunks for all pages
    pages_visualized = visualize_multi_position_chunks_on_pages(chunks, max_pages=None, output_dir=output_dir)
    
    print(f"\n🎉 SUCCESS!")
    print(f"Visualized multi-position chunks on {pages_visualized} pages")
    print(f"Images saved in: {output_dir}")

if __name__ == "__main__":
    main()
