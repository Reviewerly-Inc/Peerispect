"""
MinerU PDF Parser Test Script
Tests MinerU's PDF parsing capabilities and positional data extraction
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add the project root to the path to import our existing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tests/mineru_experiment/test_results/mineru_test.log"),
        logging.StreamHandler()
    ]
)

class MinerUTester:
    def __init__(self, test_dir: str = "tests/mineru_experiment"):
        self.test_dir = Path(test_dir)
        self.input_dir = self.test_dir / "input_pdfs"
        self.output_dir = self.test_dir / "outputs"
        self.results_dir = self.test_dir / "test_results"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # MinerU availability check
        self.mineru_available = self._check_mineru_availability()
        
    def _check_mineru_availability(self) -> bool:
        """Check if MinerU is available and install if needed."""
        try:
            import mineru
            logging.info("✅ MinerU is available")
            return True
        except ImportError:
            logging.warning("❌ MinerU not available. Installing...")
            return self._install_mineru()
    
    def _install_mineru(self) -> bool:
        """Install MinerU using pip."""
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "-U", "mineru[core]"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logging.info("✅ MinerU installed successfully")
                return True
            else:
                logging.error(f"❌ Failed to install MinerU: {result.stderr}")
                return False
        except Exception as e:
            logging.error(f"❌ Error installing MinerU: {e}")
            return False
    
    def test_basic_parsing(self, pdf_path: str) -> Dict[str, Any]:
        """Test basic PDF parsing with MinerU."""
        if not self.mineru_available:
            return {"error": "MinerU not available"}
        
        try:
            import subprocess
            import json
            
            pdf_file = Path(pdf_path)
            name_without_suff = pdf_file.stem
            
            # Create output directory for this PDF
            output_dir = self.output_dir / f"{name_without_suff}_mineru"
            output_dir.mkdir(exist_ok=True)
            
            start_time = time.time()
            
            # Use MinerU command-line interface
            # mineru -p <input_path> -o <output_path>
            cmd = [
                "mineru", 
                "-p", str(pdf_file),
                "-o", str(output_dir)
            ]
            
            logging.info(f"Running MinerU command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"MinerU command failed: {result.stderr}",
                    "elapsed_time": elapsed_time
                }
            
            # Look for output files
            md_files = list(output_dir.glob("*.md"))
            json_files = list(output_dir.glob("*.json"))
            image_dirs = [d for d in output_dir.iterdir() if d.is_dir() and "image" in d.name.lower()]
            
            # Read the generated markdown
            markdown_content = ""
            markdown_file = None
            if md_files:
                markdown_file = md_files[0]  # Take the first markdown file
                with open(markdown_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            
            # Read JSON output for positional data
            positional_data = {}
            if json_files:
                try:
                    with open(json_files[0], 'r', encoding='utf-8') as f:
                        positional_data = json.load(f)
                except Exception as e:
                    logging.warning(f"Could not read JSON output: {e}")
            
            return {
                "success": True,
                "parse_method": "mineru_cli",
                "elapsed_time": elapsed_time,
                "markdown_file": str(markdown_file) if markdown_file else None,
                "image_dir": str(image_dirs[0]) if image_dirs else None,
                "json_file": str(json_files[0]) if json_files else None,
                "markdown_length": len(markdown_content),
                "markdown_preview": markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content,
                "positional_data_available": bool(positional_data),
                "output_dir": str(output_dir)
            }
            
        except Exception as e:
            logging.error(f"Error in basic parsing: {e}")
            return {"error": str(e)}
    
    def test_positional_data(self, pdf_path: str) -> Dict[str, Any]:
        """Test extracting positional data from MinerU output."""
        if not self.mineru_available:
            return {"error": "MinerU not available"}
        
        try:
            # This would need to be implemented based on MinerU's actual output format
            # For now, we'll return a placeholder structure
            return {
                "success": True,
                "positional_data_available": True,
                "note": "Positional data extraction needs to be implemented based on MinerU's output format"
            }
        except Exception as e:
            logging.error(f"Error extracting positional data: {e}")
            return {"error": str(e)}
    
    def compare_with_existing_parsers(self, pdf_path: str) -> Dict[str, Any]:
        """Compare MinerU results with existing parsers (Docling, Nougat, PyPDF2)."""
        try:
            # Import our existing parsers
            from app.pdf_parser import PDFParser
            
            pdf_parser = PDFParser()
            comparison_results = {}
            
            # Test with existing parsers
            for method in pdf_parser.available_methods:
                try:
                    output_path = self.output_dir / f"comparison_{method}.md"
                    result = pdf_parser.parse_pdf(
                        pdf_path, 
                        str(output_path), 
                        method=method
                    )
                    
                    # Read the output
                    if output_path.exists():
                        with open(output_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        comparison_results[method] = {
                            "success": True,
                            "content_length": len(content),
                            "output_path": str(output_path)
                        }
                    else:
                        comparison_results[method] = {"success": False, "error": "No output file"}
                        
                except Exception as e:
                    comparison_results[method] = {"success": False, "error": str(e)}
            
            return comparison_results
            
        except Exception as e:
            logging.error(f"Error in comparison: {e}")
            return {"error": str(e)}
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite on all available PDFs."""
        results = {
            "test_timestamp": time.time(),
            "mineru_available": self.mineru_available,
            "test_pdfs": [],
            "summary": {}
        }
        
        if not self.mineru_available:
            results["error"] = "MinerU not available - cannot run tests"
            return results
        
        # Get all PDF files
        pdf_files = list(self.input_dir.glob("*.pdf"))
        logging.info(f"Found {len(pdf_files)} PDF files to test")
        
        for pdf_file in pdf_files:
            logging.info(f"Testing PDF: {pdf_file.name}")
            
            pdf_result = {
                "pdf_name": pdf_file.name,
                "pdf_size": pdf_file.stat().st_size,
                "basic_parsing": self.test_basic_parsing(str(pdf_file)),
                "positional_data": self.test_positional_data(str(pdf_file)),
                "comparison": self.compare_with_existing_parsers(str(pdf_file))
            }
            
            results["test_pdfs"].append(pdf_result)
        
        # Generate summary
        results["summary"] = self._generate_summary(results)
        
        # Save results
        results_file = self.results_dir / "test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Test results saved to: {results_file}")
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of test results."""
        summary = {
            "total_pdfs": len(results["test_pdfs"]),
            "successful_parses": 0,
            "failed_parses": 0,
            "average_parse_time": 0,
            "parser_comparison": {}
        }
        
        parse_times = []
        
        for pdf_result in results["test_pdfs"]:
            basic_parsing = pdf_result.get("basic_parsing", {})
            if basic_parsing.get("success"):
                summary["successful_parses"] += 1
                if "elapsed_time" in basic_parsing:
                    parse_times.append(basic_parsing["elapsed_time"])
            else:
                summary["failed_parses"] += 1
        
        if parse_times:
            summary["average_parse_time"] = sum(parse_times) / len(parse_times)
        
        return summary

def main():
    """Main function to run MinerU tests."""
    print("🧪 Starting MinerU PDF Parser Test Suite")
    print("=" * 50)
    
    tester = MinerUTester()
    
    if not tester.mineru_available:
        print("❌ MinerU is not available. Please install it first.")
        return
    
    # Run the full test suite
    results = tester.run_full_test_suite()
    
    # Print summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Total PDFs tested: {results['summary']['total_pdfs']}")
    print(f"Successful parses: {results['summary']['successful_parses']}")
    print(f"Failed parses: {results['summary']['failed_parses']}")
    print(f"Average parse time: {results['summary']['average_parse_time']:.2f}s")
    
    print(f"\n📁 Results saved to: {tester.results_dir}")
    print("✅ Test suite completed!")

if __name__ == "__main__":
    main()
