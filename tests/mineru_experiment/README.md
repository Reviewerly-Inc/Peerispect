# MinerU PDF Parser Experiment

This directory contains experiments with MinerU as a PDF parser for the Peerispect pipeline.

## Goals

1. **Better PDF Parsing**: Test MinerU's parsing quality compared to existing parsers (Docling, Nougat, PyPDF2)
2. **Positional Data**: Extract bounding box coordinates for text elements to enable frontend highlighting
3. **Integration Planning**: Prepare for integration into the main pipeline

## Directory Structure

```
tests/mineru_experiment/
├── input_pdfs/              # Test PDF files
├── outputs/                 # MinerU parsing outputs
├── test_results/            # Test results and logs
├── test_mineru_parser.py    # Main test script
├── requirements_mineru.txt  # Python dependencies
├── setup_mineru.sh         # Setup script
└── README.md               # This file
```

## Setup

### Option 1: Quick Setup (Recommended)
```bash
cd tests/mineru_experiment
./setup_mineru.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv_mineru
source venv_mineru/bin/activate

# Install dependencies
pip install -r requirements_mineru.txt
```

## Running Tests

```bash
# Activate environment
source venv_mineru/bin/activate

# Run tests
python test_mineru_parser.py
```

## Test Features

### 1. Basic Parsing Test
- Tests MinerU's core PDF parsing functionality
- Measures parsing time and quality
- Outputs markdown and extracted images

### 2. Positional Data Extraction
- Extracts bounding box coordinates for text elements
- Enables frontend highlighting capabilities
- Provides layout analysis data

### 3. Parser Comparison
- Compares MinerU with existing parsers:
  - Docling (standard)
  - Nougat (vision-based)
  - PyPDF2 (fallback)
- Measures parsing quality and speed

## Expected Outputs

### Markdown Files
- Clean markdown with proper formatting
- Preserved structure and hierarchy
- Mathematical formulas and tables

### Positional Data
- Bounding box coordinates for each text element
- Page-level positioning information
- Layout analysis results

### Images
- Extracted figures and diagrams
- High-quality image outputs
- Proper image references in markdown

## Integration Plan

1. **Phase 1**: Test and validate MinerU parsing quality
2. **Phase 2**: Implement positional data extraction
3. **Phase 3**: Create MinerU parser class for main pipeline
4. **Phase 4**: Integrate with existing chunking system
5. **Phase 5**: Add positional data to chunk metadata
6. **Phase 6**: Update frontend to use positional data

## Configuration

MinerU creates a `magic-pdf.json` configuration file in your home directory. You can modify this to:
- Enable/disable specific features
- Adjust parsing parameters
- Configure model settings

## Troubleshooting

### Common Issues

1. **Installation fails**: Try installing without the extra index URL
2. **Model download fails**: Check internet connection and try again
3. **Memory issues**: Reduce batch size in configuration
4. **CUDA issues**: Ensure proper GPU drivers and CUDA installation

### Logs

Check `test_results/mineru_test.log` for detailed logs and error messages.

## Next Steps

1. Run initial tests to validate MinerU functionality
2. Analyze parsing quality compared to existing methods
3. Implement positional data extraction
4. Plan integration with main pipeline
5. Update Docker configuration for MinerU support
