#!/usr/bin/env python3
"""
Test V5 column-aware chunking on lQgm3UvGNY.pdf
"""

import sys
from pathlib import Path
from pdf_overlay_v5 import PDFOverlayV5

def main():
    pdf_file = Path("../../outputs/pdfs/lQgm3UvGNY.pdf")
    output_folder = Path("pdf_overlay_v5_lqgm3uvgny")
    
    if not pdf_file.exists():
        print(f"❌ PDF file not found: {pdf_file}")
        return
    
    print(f"🚀 Testing V5 column-aware chunking on: {pdf_file.name}")
    print(f"📁 Output directory: {output_folder}")
    print("=" * 60)
    
    try:
        # Create overlay visualizer
        overlay_creator = PDFOverlayV5(pdf_file, output_folder)
        
        # Run full visualization
        overlay_pdf = overlay_creator.run_full_visualization()
        
        if overlay_pdf:
            print(f"\n🎉 SUCCESS! V5 column-aware overlay created for {pdf_file.name}!")
            print(f"📁 Check the output file: {overlay_pdf}")
        else:
            print("❌ Failed to create PDF overlay")
            
    except Exception as e:
        print(f"❌ Error creating PDF overlay: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
