#!/usr/bin/env python3
"""
Simple script to create PDF chunk overlays for any PDF file.
Usage: python3 create_pdf_overlay.py <pdf_path>
"""

import sys
from pathlib import Path
from pdf_chunk_overlay import PDFChunkOverlay

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 create_pdf_overlay.py <pdf_path>")
        print("Example: python3 create_pdf_overlay.py ../../outputs/pdfs/Zj8UqVxClT.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"❌ File is not a PDF: {pdf_path}")
        sys.exit(1)
    
    # Create output directory based on PDF name
    output_dir = Path(f"pdf_overlay_{pdf_path.stem}")
    
    print(f"🚀 Creating PDF chunk overlay for: {pdf_path.name}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    try:
        # Create overlay visualizer
        overlay_creator = PDFChunkOverlay(pdf_path, output_dir)
        
        # Run full visualization
        overlay_pdf, legend_pdf = overlay_creator.run_full_visualization()
        
        if overlay_pdf and legend_pdf:
            print(f"\n🎉 SUCCESS! PDF overlay created successfully!")
            print(f"📁 Check the output directory: {output_dir}")
            print(f"🎨 Overlay PDF: {overlay_pdf.name}")
            print(f"📋 Legend PDF: {legend_pdf.name}")
        else:
            print("❌ Failed to create PDF overlay")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error creating PDF overlay: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
