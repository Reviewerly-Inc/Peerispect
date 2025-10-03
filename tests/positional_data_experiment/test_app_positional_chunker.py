#!/usr/bin/env python3
"""
Test script for the app positional chunker with PDF overlay visualization.
Uses the integrated V5 positional chunker from app/4a_chunk_positional.py
"""

import os
import sys
import json
from pathlib import Path
import importlib.util

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

def test_app_positional_chunker():
    """Test the app positional chunker and create PDF overlay."""
    
    # Import the app positional chunker
    spec = importlib.util.spec_from_file_location("positional_chunker", "../../app/4a_chunk_positional.py")
    chunker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chunker_module)
    
    # Test PDF - use the available PDF
    pdf_path = "../../outputs/pdfs/odjMSBSWRt.pdf"
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found. Please ensure a PDF exists in outputs/pdfs/")
        return None
    output_dir = "app_positional_test_output"
    
    print(f"🚀 Testing app positional chunker on {pdf_path}")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Run positional chunking
    result = chunker_module.chunk_positional(
        pdf_path=pdf_path,
        output_dir=str(output_path),
        max_tokens=512,
        column_split_x=300.0
    )
    
    print(f"✅ Generated {result['num_chunks']} positional chunks")
    print(f"📁 Chunks saved to: {result['chunks_path']}")
    print(f"📄 Docling dict: {result['docling_dict_path']}")
    print(f"📝 Docling markdown: {result['docling_markdown_path']}")
    
    # Load chunks for analysis
    with open(result['chunks_path'], 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Analyze chunks
    chunks_with_multiple_pos = sum(1 for c in chunks if len(c['positions']) > 1)
    chunks_spanning_cols = sum(1 for c in chunks if c['spans_columns'])
    chunks_spanning_pages = sum(1 for c in chunks if c['spans_pages'])
    
    print(f"\n📊 CHUNK ANALYSIS:")
    print(f"  • Total chunks: {len(chunks)}")
    print(f"  • Chunks with multiple positions: {chunks_with_multiple_pos}")
    print(f"  • Chunks spanning columns: {chunks_spanning_cols}")
    print(f"  • Chunks spanning pages: {chunks_spanning_pages}")
    print(f"  • Average tokens per chunk: {sum(c['tokens'] for c in chunks) / len(chunks):.1f}")
    
    # Show sample chunks
    print(f"\n📋 SAMPLE CHUNKS:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"  {i+1}. {chunk['id']} ({chunk['tokens']} tokens)")
        print(f"     Pages: {chunk['pages']}, Positions: {len(chunk['positions'])}")
        print(f"     Spans columns: {chunk['spans_columns']}, Spans pages: {chunk['spans_pages']}")
        print(f"     Text: {chunk['text'][:100]}...")
        print()
    
    # Create PDF overlay if visualization dependencies are available
    try:
        create_pdf_overlay(pdf_path, result['chunks_path'], output_path)
    except ImportError as e:
        print(f"⚠️  PDF overlay visualization not available: {e}")
        print("   Install with: pip install PyPDF2 reportlab")
    
    return result

def create_pdf_overlay(pdf_path, chunks_path, output_dir):
    """Create PDF overlay with chunk highlights."""
    try:
        import io
        from PyPDF2 import PdfReader, PdfWriter
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.colors import Color
        from reportlab.lib.units import inch
        PYPDF2_AVAILABLE = True
        REPORTLAB_AVAILABLE = True
    except ImportError as e:
        raise ImportError(f"Required libraries not available: {e}")
    
    print("🎨 Creating PDF overlay with chunk highlights...")
    
    # Load chunks
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Read the original PDF
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    # Group chunks by page
    page_chunks = {}
    for chunk in chunks:
        for pos in chunk.get('positions', []):
            page_num = pos['page']
            if page_num not in page_chunks:
                page_chunks[page_num] = []
            page_chunks[page_num].append((chunk, pos))
    
    # Generate colors for chunks
    import colorsys
    chunk_colors = []
    for i in range(len(chunks)):
        hue = (i * 137.5) % 360
        saturation = 0.6 + (i % 3) * 0.1
        value = 0.7 + (i % 2) * 0.1
        rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
        chunk_colors.append(Color(rgb[0], rgb[1], rgb[2], alpha=0.4))
    
    chunk_id_to_color = {}
    for i, chunk in enumerate(chunks):
        chunk_id_to_color[chunk['id']] = chunk_colors[i]
    
    # Process each page
    for page_num in range(len(reader.pages)):
        print(f"  📄 Processing page {page_num + 1}...")
        
        # Get the original page
        original_page = reader.pages[page_num]
        
        # Get page dimensions
        page_width = float(original_page.mediabox.width)
        page_height = float(original_page.mediabox.height)
        
        # Create overlay canvas
        overlay_buffer = io.BytesIO()
        overlay_canvas = canvas.Canvas(overlay_buffer, pagesize=(page_width, page_height))
        
        # Draw chunk highlights if this page has chunks
        if (page_num + 1) in page_chunks:
            chunks_on_page = page_chunks[page_num + 1]
            
            for chunk, pos in chunks_on_page:
                bbox = pos['bounding_box']
                chunk_id = chunk['id']
                
                # Get color for this chunk
                color = chunk_id_to_color.get(chunk_id, Color(0.5, 0.5, 0.5, 0.4))
                
                # Draw rectangle (PDF coordinates: bottom-left origin)
                overlay_canvas.setFillColor(color)
                overlay_canvas.rect(
                    bbox['left'],
                    bbox['bottom'],
                    bbox['right'] - bbox['left'],
                    bbox['top'] - bbox['bottom'],
                    fill=True,
                    stroke=False
                )
                
                # Add chunk label
                overlay_canvas.setFillColor(Color(0, 0, 0, 0.8))
                overlay_canvas.setFont("Helvetica-Bold", 7)
                
                # Position label at top-left of chunk
                label_x = bbox['left'] + 2
                label_y = bbox['top'] - 8
                
                # Truncate chunk ID if too long
                label_text = chunk_id[:15] + "..." if len(chunk_id) > 15 else chunk_id
                overlay_canvas.drawString(label_x, label_y, label_text)
                
                # Add position info
                overlay_canvas.setFont("Helvetica", 5)
                pos_info = f"P{pos['page']} {pos['column']}"
                overlay_canvas.drawString(label_x, label_y - 6, pos_info)
            
            # Add page info
            overlay_canvas.setFillColor(Color(0, 0, 0, 0.6))
            overlay_canvas.setFont("Helvetica", 8)
            overlay_canvas.drawString(10, page_height - 15, f"Page {page_num + 1} - {len(chunks_on_page)} chunk positions (App V5)")
            
            # Add column separator line
            overlay_canvas.setStrokeColor(Color(1, 0, 0, 0.3))
            overlay_canvas.setLineWidth(0.5)
            overlay_canvas.line(page_width/2, 0, page_width/2, page_height)
        
        # Finish the overlay page
        overlay_canvas.showPage()
        overlay_canvas.save()
        
        # Get the overlay page
        overlay_buffer.seek(0)
        overlay_page = PdfReader(overlay_buffer).pages[0]
        
        # Merge overlay with original page
        original_page.merge_page(overlay_page)
        
        # Add to writer
        writer.add_page(original_page)
    
    # Write the output PDF
    output_pdf = output_dir / "app_positional_chunks_overlay.pdf"
    with open(output_pdf, 'wb') as output_file:
        writer.write(output_file)
    
    print(f"✅ Created PDF overlay: {output_pdf}")
    return output_pdf

if __name__ == "__main__":
    test_app_positional_chunker()
