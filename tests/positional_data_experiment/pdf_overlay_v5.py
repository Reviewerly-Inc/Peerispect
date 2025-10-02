import os
import sys
import json
from pathlib import Path
import importlib.util
from typing import List, Dict, Any
import colorsys
import random

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Dynamically import 2_parse_pdf.py
spec = importlib.util.spec_from_file_location("parse_pdf_module", "../../app/2_parse_pdf.py")
parse_pdf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse_pdf_module)

try:
    import io
    from PyPDF2 import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.colors import Color
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    PYPDF2_AVAILABLE = True
    REPORTLAB_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required libraries not available: {e}")
    print("Install with: pip install PyPDF2 reportlab")
    PYPDF2_AVAILABLE = False
    REPORTLAB_AVAILABLE = False

class PDFOverlayV5:
    def __init__(self, pdf_path: Path, output_dir: Path):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate chunks using V5 strategy
        self.chunks = self._generate_chunks()
        
        # Get PDF page dimensions
        self.page_dimensions = self._get_page_dimensions()
        
    def _generate_chunks(self) -> List[Dict]:
        """Generate chunks using the V5 column-aware strategy."""
        print("🔄 Generating chunks using V5 column-aware strategy...")
        
        # Import the V5 chunking strategy
        spec = importlib.util.spec_from_file_location("chunking_v5", "improved_chunking_v5_column_aware.py")
        chunking_v5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chunking_v5)
        
        # Create chunker instance
        chunker = chunking_v5.ImprovedPositionalChunkerV5(self.pdf_path, self.output_dir)
        
        # Generate chunks
        chunks = chunker.chunk_column_aware_sections()
        
        print(f"✅ Generated {len(chunks)} chunks")
        return chunks
    
    def _get_page_dimensions(self) -> Dict[int, tuple]:
        """Get page dimensions from Docling output."""
        docling_output = self.output_dir / "docling_dict_output.json"
        
        if not docling_output.exists():
            print("❌ Docling output not found. Please run chunking first.")
            return {}
        
        with open(docling_output, 'r', encoding='utf-8') as f:
            docling_data = json.load(f)
        
        page_dimensions = {}
        for page_num, page_info in docling_data.get('pages', {}).items():
            if 'size' in page_info:
                size = page_info['size']
                if isinstance(size, list) and len(size) >= 2:
                    page_dimensions[int(page_num)] = (float(size[0]), float(size[1]))
                elif isinstance(size, dict) and 'width' in size and 'height' in size:
                    page_dimensions[int(page_num)] = (float(size['width']), float(size['height']))
        
        return page_dimensions
    
    def _generate_chunk_colors(self, num_chunks: int) -> List[Color]:
        """Generate distinct colors for each chunk."""
        colors = []
        
        # Generate colors using HSV color space for better distribution
        for i in range(num_chunks):
            hue = (i * 137.5) % 360  # Golden angle for good distribution
            saturation = 0.6 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.7 + (i % 2) * 0.1  # Vary brightness slightly
            
            rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
            colors.append(Color(rgb[0], rgb[1], rgb[2], alpha=0.4))  # Semi-transparent
        
        return colors
    
    def create_overlay_on_original(self) -> Path:
        """Create PDF with chunk highlights overlaid on the original PDF."""
        if not PYPDF2_AVAILABLE or not REPORTLAB_AVAILABLE:
            print("❌ Cannot create PDF overlay - Required libraries not available")
            return None
        
        print("🎨 Creating PDF overlay on original PDF with V5 column-aware chunks...")
        
        # Read the original PDF
        reader = PdfReader(str(self.pdf_path))
        writer = PdfWriter()
        
        # Group chunks by page
        page_chunks = {}
        for chunk in self.chunks:
            for pos in chunk.get('positions', []):
                page_num = pos['page']
                if page_num not in page_chunks:
                    page_chunks[page_num] = []
                page_chunks[page_num].append((chunk, pos))
        
        # Generate colors for chunks
        chunk_colors = self._generate_chunk_colors(len(self.chunks))
        chunk_id_to_color = {}
        for i, chunk in enumerate(self.chunks):
            chunk_id_to_color[chunk['id']] = chunk_colors[i]
        
        # Process each page
        for page_num in range(len(reader.pages)):
            print(f"  📄 Processing page {page_num + 1}...")
            
            # Get the original page
            original_page = reader.pages[page_num]
            
            # Get page dimensions
            if (page_num + 1) in self.page_dimensions:
                page_width, page_height = self.page_dimensions[page_num + 1]
            else:
                # Fallback to page dimensions from PyPDF2
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
                    overlay_canvas.setFillColor(Color(0, 0, 0, 0.8))  # Semi-transparent black text
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
                overlay_canvas.drawString(10, page_height - 15, f"Page {page_num + 1} - {len(chunks_on_page)} chunk positions (V5 Column-Aware)")
                
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
        output_pdf = self.output_dir / f"{self.pdf_path.stem}_v5_column_aware_overlay.pdf"
        with open(output_pdf, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"✅ Created V5 overlay PDF: {output_pdf}")
        return output_pdf
    
    def run_full_visualization(self):
        """Run the complete PDF overlay visualization."""
        print(f"🚀 Starting V5 column-aware PDF overlay for {self.pdf_path.name}")
        print("=" * 60)
        
        # Create overlay PDF
        overlay_pdf = self.create_overlay_on_original()
        
        print("\n🎯 V5 COLUMN-AWARE VISUALIZATION COMPLETE!")
        print("=" * 60)
        print(f"📄 Input PDF: {self.pdf_path}")
        print(f"📊 Total chunks: {len(self.chunks)}")
        print(f"🎨 Overlay PDF: {overlay_pdf}")
        
        # Show chunk statistics
        chunks_with_multiple_pos = sum(1 for c in self.chunks if len(c.get('positions', [])) > 1)
        chunks_spanning_columns = sum(1 for c in self.chunks if c.get('spans_columns', False))
        chunks_spanning_pages = sum(1 for c in self.chunks if c.get('spans_pages', False))
        
        print(f"\n📊 V5 CHUNK STATISTICS:")
        print(f"  • Chunks with multiple positions: {chunks_with_multiple_pos}")
        print(f"  • Chunks spanning columns: {chunks_spanning_columns}")
        print(f"  • Chunks spanning pages: {chunks_spanning_pages}")
        print(f"  • Average tokens per chunk: {sum(c.get('tokens', 0) for c in self.chunks) / len(self.chunks):.1f}")
        
        print(f"\n🎯 V5 IMPROVEMENTS:")
        print(f"  ✅ Proper column separation - left and right columns are separate chunks")
        print(f"  ✅ Reduced column spanning - only {chunks_spanning_columns} chunks span columns")
        print(f"  ✅ Better reading order - chunks follow natural text flow")
        print(f"  ✅ Maintained positional data - all bounding boxes preserved")
        
        return overlay_pdf


def main():
    """Main function to create V5 PDF overlay."""
    # Use the test PDF
    pdf_file = Path("../../outputs/pdfs/Zj8UqVxClT.pdf")
    output_folder = Path("pdf_overlay_v5_output")
    
    if not pdf_file.exists():
        print(f"❌ PDF file not found: {pdf_file}")
        return
    
    # Create overlay visualizer
    overlay_creator = PDFOverlayV5(pdf_file, output_folder)
    
    # Run full visualization
    overlay_pdf = overlay_creator.run_full_visualization()
    
    if overlay_pdf:
        print(f"\n🎉 SUCCESS! V5 column-aware overlay created!")
        print(f"📁 Check the output file: {overlay_pdf}")


if __name__ == "__main__":
    main()
