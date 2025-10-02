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
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.colors import Color
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("❌ ReportLab not available. Install with: pip install reportlab")

class PDFChunkOverlay:
    def __init__(self, pdf_path: Path, output_dir: Path):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate chunks using V4 strategy
        self.chunks = self._generate_chunks()
        
        # Get PDF page dimensions
        self.page_dimensions = self._get_page_dimensions()
        
    def _generate_chunks(self) -> List[Dict]:
        """Generate chunks using the V4 strategy."""
        print("🔄 Generating chunks using V4 strategy...")
        
        # Import the V4 chunking strategy
        spec = importlib.util.spec_from_file_location("chunking_v4", "improved_chunking_v4.py")
        chunking_v4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chunking_v4)
        
        # Create chunker instance
        chunker = chunking_v4.ImprovedPositionalChunkerV4(self.pdf_path, self.output_dir)
        
        # Generate chunks
        chunks = chunker.chunk_by_sections_with_all_positions()
        
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
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (i % 2) * 0.1  # Vary brightness slightly
            
            rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
            colors.append(Color(rgb[0], rgb[1], rgb[2], alpha=0.3))  # Semi-transparent
        
        return colors
    
    def create_overlay_pdf(self) -> Path:
        """Create PDF overlay with chunk highlights."""
        if not REPORTLAB_AVAILABLE:
            print("❌ Cannot create PDF overlay - ReportLab not available")
            return None
        
        print("🎨 Creating PDF overlay with chunk highlights...")
        
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
        
        # Create output PDF
        output_pdf = self.output_dir / f"{self.pdf_path.stem}_chunk_overlay.pdf"
        c = canvas.Canvas(str(output_pdf))
        
        # Process each page
        for page_num in sorted(page_chunks.keys()):
            print(f"  📄 Processing page {page_num}...")
            
            # Get page dimensions
            if page_num in self.page_dimensions:
                page_width, page_height = self.page_dimensions[page_num]
            else:
                page_width, page_height = 612, 792  # Default A4 size
            
            # Set page size
            c.setPageSize((page_width, page_height))
            
            # Draw chunk highlights
            chunks_on_page = page_chunks[page_num]
            for chunk, pos in chunks_on_page:
                bbox = pos['bounding_box']
                chunk_id = chunk['id']
                
                # Get color for this chunk
                color = chunk_id_to_color.get(chunk_id, Color(0.5, 0.5, 0.5, 0.3))
                
                # Draw rectangle (PDF coordinates: bottom-left origin)
                c.setFillColor(color)
                c.rect(
                    bbox['left'],
                    bbox['bottom'],
                    bbox['right'] - bbox['left'],
                    bbox['top'] - bbox['bottom'],
                    fill=True,
                    stroke=False
                )
                
                # Add chunk label
                c.setFillColor(Color(0, 0, 0, 1))  # Black text
                c.setFont("Helvetica-Bold", 8)
                
                # Position label at top-left of chunk
                label_x = bbox['left'] + 2
                label_y = bbox['top'] - 10
                
                # Truncate chunk ID if too long
                label_text = chunk_id[:20] + "..." if len(chunk_id) > 20 else chunk_id
                c.drawString(label_x, label_y, label_text)
                
                # Add position info
                c.setFont("Helvetica", 6)
                pos_info = f"P{pos['page']} {pos['column']}"
                c.drawString(label_x, label_y - 8, pos_info)
            
            # Add page info
            c.setFillColor(Color(0, 0, 0, 0.7))
            c.setFont("Helvetica", 10)
            c.drawString(10, page_height - 20, f"Page {page_num} - {len(chunks_on_page)} chunk positions")
            
            # Add column separator line
            c.setStrokeColor(Color(1, 0, 0, 0.5))
            c.setLineWidth(1)
            c.line(page_width/2, 0, page_width/2, page_height)
            
            c.showPage()
        
        c.save()
        
        print(f"✅ Created overlay PDF: {output_pdf}")
        return output_pdf
    
    def create_legend_pdf(self) -> Path:
        """Create a legend PDF showing chunk colors and info."""
        if not REPORTLAB_AVAILABLE:
            return None
        
        print("📋 Creating chunk legend...")
        
        legend_pdf = self.output_dir / f"{self.pdf_path.stem}_chunk_legend.pdf"
        c = canvas.Canvas(str(legend_pdf))
        
        # Set page size
        c.setPageSize(letter)
        page_width, page_height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, page_height - 50, f"Chunk Legend: {self.pdf_path.name}")
        
        # Chunk info
        c.setFont("Helvetica", 12)
        c.drawString(50, page_height - 80, f"Total Chunks: {len(self.chunks)}")
        
        # Generate colors
        chunk_colors = self._generate_chunk_colors(len(self.chunks))
        
        # Draw legend items
        y_position = page_height - 120
        items_per_row = 2
        item_width = (page_width - 100) / items_per_row
        
        for i, chunk in enumerate(self.chunks):
            if i > 0 and i % items_per_row == 0:
                y_position -= 60  # Move to next row
            
            x_position = 50 + (i % items_per_row) * item_width
            
            # Draw color box
            color = chunk_colors[i]
            c.setFillColor(color)
            c.rect(x_position, y_position - 20, 30, 20, fill=True, stroke=True)
            
            # Draw chunk info
            c.setFillColor(Color(0, 0, 0, 1))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x_position + 35, y_position - 5, chunk['id'][:30])
            
            c.setFont("Helvetica", 8)
            c.drawString(x_position + 35, y_position - 15, f"Tokens: {chunk.get('tokens', 0)}")
            c.drawString(x_position + 35, y_position - 25, f"Pages: {chunk.get('pages', [])}")
            
            # Check if chunk spans columns
            spans_columns = chunk.get('spans_columns', False)
            if spans_columns:
                c.setFillColor(Color(1, 0, 0, 1))
                c.drawString(x_position + 35, y_position - 35, "SPANS COLUMNS")
        
        c.save()
        
        print(f"✅ Created legend PDF: {legend_pdf}")
        return legend_pdf
    
    def run_full_visualization(self):
        """Run the complete PDF overlay visualization."""
        print(f"🚀 Starting PDF chunk overlay visualization for {self.pdf_path.name}")
        print("=" * 60)
        
        # Create overlay PDF
        overlay_pdf = self.create_overlay_pdf()
        
        # Create legend PDF
        legend_pdf = self.create_legend_pdf()
        
        print("\n🎯 VISUALIZATION COMPLETE!")
        print("=" * 60)
        print(f"📄 Input PDF: {self.pdf_path}")
        print(f"📊 Total chunks: {len(self.chunks)}")
        print(f"🎨 Overlay PDF: {overlay_pdf}")
        print(f"📋 Legend PDF: {legend_pdf}")
        
        # Show chunk statistics
        chunks_with_multiple_pos = sum(1 for c in self.chunks if len(c.get('positions', [])) > 1)
        chunks_spanning_columns = sum(1 for c in self.chunks if c.get('spans_columns', False))
        
        print(f"\n📊 CHUNK STATISTICS:")
        print(f"  • Chunks with multiple positions: {chunks_with_multiple_pos}")
        print(f"  • Chunks spanning columns: {chunks_spanning_columns}")
        print(f"  • Average tokens per chunk: {sum(c.get('tokens', 0) for c in self.chunks) / len(self.chunks):.1f}")
        
        return overlay_pdf, legend_pdf


def main():
    """Main function to create PDF chunk overlay."""
    # Use the test PDF
    pdf_file = Path("../../outputs/pdfs/Zj8UqVxClT.pdf")
    output_folder = Path("pdf_chunk_overlay_output")
    
    if not pdf_file.exists():
        print(f"❌ PDF file not found: {pdf_file}")
        return
    
    # Create overlay visualizer
    overlay_creator = PDFChunkOverlay(pdf_file, output_folder)
    
    # Run full visualization
    overlay_pdf, legend_pdf = overlay_creator.run_full_visualization()
    
    if overlay_pdf and legend_pdf:
        print(f"\n🎉 SUCCESS! Check the output files:")
        print(f"  📁 Output directory: {output_folder}")
        print(f"  🎨 Overlay: {overlay_pdf.name}")
        print(f"  📋 Legend: {legend_pdf.name}")


if __name__ == "__main__":
    main()
