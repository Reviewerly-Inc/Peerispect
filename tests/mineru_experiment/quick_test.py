#!/usr/bin/env python3
"""
Quick test script to verify MinerU setup
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

def test_imports():
    """Test if we can import our modules."""
    print("🧪 Testing imports...")
    
    try:
        # Import using importlib since the module name starts with a number
        import importlib.util
        spec = importlib.util.spec_from_file_location("parse_pdf", "../../app/2_parse_pdf.py")
        parse_pdf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parse_pdf_module)
        print("✅ Existing PDF parser imported successfully")
    except Exception as e:
        print(f"❌ Failed to import existing PDF parser: {e}")
        return False
    
    try:
        from mineru_parser import MinerUParser
        print("✅ MinerU parser imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MinerU parser: {e}")
        return False
    
    return True

def test_mineru_availability():
    """Test if MinerU is available."""
    print("\n🔍 Testing MinerU availability...")
    
    try:
        import mineru
        print("✅ MinerU is available")
        return True
    except ImportError:
        print("❌ MinerU not available - will need to install")
        print("   Run: pip install mineru[core]")
        return False

def test_pdf_files():
    """Test if we have PDF files to work with."""
    print("\n📄 Testing PDF files...")
    
    input_dir = Path("input_pdfs")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if pdf_files:
        print(f"✅ Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            print(f"   - {pdf.name} ({size_mb:.1f} MB)")
        return True
    else:
        print("❌ No PDF files found in input_pdfs/")
        return False

def test_existing_parsers():
    """Test existing parsers."""
    print("\n🔧 Testing existing parsers...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("parse_pdf", "../../app/2_parse_pdf.py")
        parse_pdf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parse_pdf_module)
        
        # Create a PDFParser instance
        parser = parse_pdf_module.PDFParser()
        print(f"✅ Available parsers: {parser.available_methods}")
        return True
    except Exception as e:
        print(f"❌ Error testing existing parsers: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 MinerU Experiment Quick Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_mineru_availability,
        test_pdf_files,
        test_existing_parsers
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed! Ready to run MinerU experiments.")
        print("\nNext steps:")
        print("1. Install MinerU: ./setup_mineru.sh")
        print("2. Run full tests: python test_mineru_parser.py")
    else:
        print("❌ Some tests failed. Please fix issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
