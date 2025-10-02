#!/usr/bin/env python3
"""
Script to overlay chunk highlights directly on the original PDF.
Usage: python3 overlay_chunks_on_pdf.py <pdf_path>
"""

import sys
from pathlib import Path
from pdf_overlay_on_original import PDFOverlayOnOriginal

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 overlay_chunks_on_pdf.py <pdf_path>")
        print("Example: python3 overlay_chunks_on_pdf.py ../../outputs/pdfs/Zj8UqVxClT.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"❌ File is not a PDF: {pdf_path}")
        sys.exit(1)
    
    # Create output directory based on PDF name
    output_dir = Path(f"pdf_overlay_on_original_{pdf_path.stem}")
    
    print(f"🚀 Creating chunk overlay on original PDF: {pdf_path.name}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    try:
        # Create overlay visualizer
        overlay_creator = PDFOverlayOnOriginal(pdf_path, output_dir)
        
        # Run full visualization
        overlay_pdf, legend_pdf = overlay_creator.run_full_visualization()
        
        if overlay_pdf and legend_pdf:
            print(f"\n🎉 SUCCESS! PDF with chunk overlay created!")
            print(f"📁 Check the output directory: {output_dir}")
            print(f"🎨 Overlay PDF: {overlay_pdf.name}")
            print(f"📋 Legend PDF: {legend_pdf.name}")
            print(f"\n💡 The overlay PDF contains the original content with colored chunk highlights!")
        else:
            print("❌ Failed to create PDF overlay")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error creating PDF overlay: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
