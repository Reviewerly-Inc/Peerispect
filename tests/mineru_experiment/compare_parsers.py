#!/usr/bin/env python3
"""
Comprehensive comparison between MinerU and Docling PDF parsers
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

class ParserComparison:
    def __init__(self, test_dir: str = "."):
        self.test_dir = Path(test_dir)
        self.input_dir = self.test_dir / "input_pdfs"
        self.output_dir = self.test_dir / "outputs"
        
    def load_existing_parser(self):
        """Load the existing PDF parser from the main app."""
        import importlib.util
        spec = importlib.util.spec_from_file_location('parse_pdf', '../../app/2_parse_pdf.py')
        parse_pdf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parse_pdf_module)
        return parse_pdf_module
    
    def test_docling_parsing(self, pdf_path: str) -> Dict[str, Any]:
        """Test Docling parsing."""
        try:
            parse_pdf_module = self.load_existing_parser()
            
            start_time = time.time()
            result = parse_pdf_module.parse_pdf_to_markdown(
                pdf_path,
                str(self.output_dir / "docling_comparison.md"),
                method="docling_standard",
                code_enrichment=True,
                formula_enrichment=True
            )
            elapsed_time = time.time() - start_time
            
            # Read the output
            markdown_content = ""
            if os.path.exists(result['output_path']):
                with open(result['output_path'], 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            
            return {
                'success': True,
                'method': 'docling_standard',
                'elapsed_time': elapsed_time,
                'output_path': result['output_path'],
                'markdown_length': len(markdown_content),
                'markdown_content': markdown_content,
                'positional_data': None,  # Docling doesn't provide positional data
                'images_extracted': False,  # Docling doesn't extract images separately
                'formulas_detected': markdown_content.count('$$') // 2,
                'tables_detected': markdown_content.count('|') // 3,  # Rough estimate
                'headers_detected': markdown_content.count('#'),
                'links_detected': markdown_content.count('[') - markdown_content.count('![')
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_mineru_parsing(self, pdf_path: str) -> Dict[str, Any]:
        """Test MinerU parsing."""
        try:
            import subprocess
            
            # Create output directory
            mineru_output_dir = self.output_dir / "mineru_comparison"
            mineru_output_dir.mkdir(exist_ok=True)
            
            start_time = time.time()
            
            # Run MinerU
            cmd = ["mineru", "-p", pdf_path, "-o", str(mineru_output_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            elapsed_time = time.time() - start_time
            
            if result.returncode != 0:
                return {'success': False, 'error': f"MinerU failed: {result.stderr}"}
            
            # Find output files
            pdf_name = Path(pdf_path).stem
            output_subdir = mineru_output_dir / pdf_name / "auto"
            
            md_files = list(output_subdir.glob("*.md"))
            json_files = list(output_subdir.glob("*_middle.json"))
            image_dirs = [d for d in output_subdir.iterdir() if d.is_dir() and "image" in d.name.lower()]
            
            # Read markdown
            markdown_content = ""
            if md_files:
                with open(md_files[0], 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            
            # Read positional data
            positional_data = {}
            if json_files:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    positional_data = json.load(f)
            
            # Count images
            image_count = 0
            if image_dirs:
                image_count = len(list(image_dirs[0].glob("*.jpg")))
            
            return {
                'success': True,
                'method': 'mineru_cli',
                'elapsed_time': elapsed_time,
                'output_path': str(md_files[0]) if md_files else None,
                'markdown_length': len(markdown_content),
                'markdown_content': markdown_content,
                'positional_data': positional_data,
                'images_extracted': image_count,
                'formulas_detected': markdown_content.count('$$') // 2,
                'tables_detected': markdown_content.count('|') // 3,  # Rough estimate
                'headers_detected': markdown_content.count('#'),
                'links_detected': markdown_content.count('[') - markdown_content.count('!['),
                'positional_elements': len(positional_data.get('pdf_info', [{}])[0].get('preproc_blocks', [])) if positional_data else 0
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_positional_data(self, positional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze MinerU's positional data structure."""
        if not positional_data:
            return {'available': False}
        
        try:
            pdf_info = positional_data.get('pdf_info', [])
            if not pdf_info:
                return {'available': False, 'error': 'No PDF info found'}
            
            first_page = pdf_info[0]
            blocks = first_page.get('preproc_blocks', [])
            
            # Analyze block types
            block_types = {}
            total_blocks = len(blocks)
            
            for block in blocks:
                block_type = block.get('type', 'unknown')
                block_types[block_type] = block_types.get(block_type, 0) + 1
            
            # Analyze bounding boxes
            bbox_stats = {
                'total_blocks': total_blocks,
                'blocks_with_bbox': sum(1 for block in blocks if 'bbox' in block),
                'avg_bbox_area': 0
            }
            
            if blocks:
                areas = []
                for block in blocks:
                    if 'bbox' in block:
                        bbox = block['bbox']
                        if len(bbox) == 4:
                            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            areas.append(area)
                
                if areas:
                    bbox_stats['avg_bbox_area'] = sum(areas) / len(areas)
            
            return {
                'available': True,
                'total_blocks': total_blocks,
                'block_types': block_types,
                'bbox_stats': bbox_stats,
                'sample_blocks': blocks[:3]  # First 3 blocks as sample
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def compare_outputs(self, docling_result: Dict[str, Any], mineru_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare the outputs of both parsers."""
        comparison = {
            'parsing_success': {
                'docling': docling_result.get('success', False),
                'mineru': mineru_result.get('success', False)
            },
            'performance': {
                'docling_time': docling_result.get('elapsed_time', 0),
                'mineru_time': mineru_result.get('elapsed_time', 0),
                'faster_parser': 'mineru' if mineru_result.get('elapsed_time', 0) < docling_result.get('elapsed_time', 0) else 'docling'
            },
            'output_quality': {
                'docling_length': docling_result.get('markdown_length', 0),
                'mineru_length': mineru_result.get('markdown_length', 0),
                'length_ratio': mineru_result.get('markdown_length', 0) / max(docling_result.get('markdown_length', 1), 1)
            },
            'features': {
                'docling_formulas': docling_result.get('formulas_detected', 0),
                'mineru_formulas': mineru_result.get('formulas_detected', 0),
                'docling_tables': docling_result.get('tables_detected', 0),
                'mineru_tables': mineru_result.get('tables_detected', 0),
                'docling_headers': docling_result.get('headers_detected', 0),
                'mineru_headers': mineru_result.get('headers_detected', 0),
                'mineru_images': mineru_result.get('images_extracted', 0),
                'mineru_positional_elements': mineru_result.get('positional_elements', 0)
            },
            'unique_features': {
                'docling_positional_data': docling_result.get('positional_data') is not None,
                'mineru_positional_data': mineru_result.get('positional_data') is not None,
                'docling_image_extraction': docling_result.get('images_extracted', False),
                'mineru_image_extraction': mineru_result.get('images_extracted', False)
            }
        }
        
        return comparison
    
    def run_comparison(self, pdf_path: str) -> Dict[str, Any]:
        """Run complete comparison between parsers."""
        print(f"🔍 Comparing parsers on: {Path(pdf_path).name}")
        print("=" * 60)
        
        # Test both parsers
        print("📄 Testing Docling...")
        docling_result = self.test_docling_parsing(pdf_path)
        
        print("⚡ Testing MinerU...")
        mineru_result = self.test_mineru_parsing(pdf_path)
        
        # Analyze positional data
        positional_analysis = self.analyze_positional_data(mineru_result.get('positional_data', {}))
        
        # Compare results
        comparison = self.compare_outputs(docling_result, mineru_result)
        
        # Create comprehensive report
        report = {
            'pdf_file': Path(pdf_path).name,
            'pdf_size_mb': Path(pdf_path).stat().st_size / (1024 * 1024),
            'docling_result': docling_result,
            'mineru_result': mineru_result,
            'positional_analysis': positional_analysis,
            'comparison': comparison,
            'timestamp': time.time()
        }
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the comparison."""
        print("\n" + "=" * 60)
        print("📊 PARSER COMPARISON SUMMARY")
        print("=" * 60)
        
        # Basic info
        print(f"📄 PDF: {report['pdf_file']} ({report['pdf_size_mb']:.1f} MB)")
        
        # Success rates
        success = report['comparison']['parsing_success']
        print(f"\n✅ Success Rates:")
        print(f"   Docling: {'✅' if success['docling'] else '❌'}")
        print(f"   MinerU:  {'✅' if success['mineru'] else '❌'}")
        
        # Performance
        perf = report['comparison']['performance']
        print(f"\n⏱️ Performance:")
        print(f"   Docling: {perf['docling_time']:.1f}s")
        print(f"   MinerU:  {perf['mineru_time']:.1f}s")
        print(f"   Faster:  {perf['faster_parser']}")
        
        # Output quality
        quality = report['comparison']['output_quality']
        print(f"\n📝 Output Quality:")
        print(f"   Docling length: {quality['docling_length']:,} chars")
        print(f"   MinerU length:  {quality['mineru_length']:,} chars")
        print(f"   Length ratio:   {quality['length_ratio']:.2f}x")
        
        # Features
        features = report['comparison']['features']
        print(f"\n🔧 Features Detected:")
        print(f"   Formulas - Docling: {features['docling_formulas']}, MinerU: {features['mineru_formulas']}")
        print(f"   Tables   - Docling: {features['docling_tables']}, MinerU: {features['mineru_tables']}")
        print(f"   Headers  - Docling: {features['docling_headers']}, MinerU: {features['mineru_headers']}")
        print(f"   Images   - MinerU:  {features['mineru_images']}")
        
        # Unique features
        unique = report['comparison']['unique_features']
        print(f"\n🌟 Unique Features:")
        print(f"   Positional Data: {'✅ MinerU' if unique['mineru_positional_data'] else '❌ None'}")
        print(f"   Image Extraction: {'✅ MinerU' if unique['mineru_image_extraction'] else '❌ None'}")
        
        # Positional data analysis
        if report['positional_analysis']['available']:
            pos = report['positional_analysis']
            print(f"\n📍 Positional Data Analysis:")
            print(f"   Total elements: {pos['total_blocks']}")
            print(f"   Block types: {pos['block_types']}")
            print(f"   Elements with bbox: {pos['bbox_stats']['blocks_with_bbox']}")

def main():
    """Run the comparison."""
    print("🚀 Starting PDF Parser Comparison")
    print("=" * 60)
    
    comparator = ParserComparison()
    
    # Test with the first PDF
    pdf_files = list(comparator.input_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in input_pdfs/")
        return
    
    pdf_path = pdf_files[0]  # Use first PDF
    report = comparator.run_comparison(str(pdf_path))
    
    # Print summary
    comparator.print_summary(report)
    
    # Save detailed report
    report_file = comparator.test_dir / "test_results" / "parser_comparison.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Detailed report saved to: {report_file}")
    print("✅ Comparison complete!")

if __name__ == "__main__":
    main()
