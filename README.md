# OpenReview Paper Processing App

This app processes OpenReview papers and reviews through a comprehensive pipeline that extracts claims from reviews and verifies them against the paper content.

## Features

- **OpenReview Crawling**: Extract papers and reviews from OpenReview URLs
- **PDF Parsing**: Multiple parsing methods (Docling, Nougat, PyPDF2 fallback)
- **Markdown Cleaning**: Remove artifacts and format content
- **Text Chunking**: Split papers into manageable chunks for retrieval
- **Review Processing**: Extract structured reviews from OpenReview data
- **Claim Extraction**: Extract claims from reviews using multiple methods (FENICE, rule-based)
- **Evidence Retrieval**: Find relevant evidence for claims using TF-IDF, BM25, or SBERT
- **Claim Verification**: Verify claims against evidence using Ollama LLMs

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama (for claim verification):
```bash
# Follow instructions at https://ollama.ai/
# Then pull a model:
ollama pull qwen3:14b
```

## Usage

### Basic Usage

```bash
python main.py "https://openreview.net/forum?id=YOUR_SUBMISSION_ID"
```

### With Configuration

```bash
python main.py "https://openreview.net/forum?id=YOUR_SUBMISSION_ID" --config config.json
```

### With OpenReview Credentials

```bash
python main.py "https://openreview.net/forum?id=YOUR_SUBMISSION_ID" --username your_email --password your_password
```

### Processing Individual Reviews

Process a single review file independently:

```bash
python process_individual_review.py path/to/review.json --chunks path/to/chunks.jsonl --output results/
```

Process all review files in a directory:

```bash
python process_individual_review.py reviews/ --directory --chunks path/to/chunks.jsonl --submission-id SUBMISSION_ID
```

## Configuration

The app uses a JSON configuration file (`config.json`) with the following options:

```json
{
    "pdf_parser": "auto",
    "parser_kwargs": {
        "code_enrichment": false,
        "formula_enrichment": false,
        "model": "0.1.0-small"
    },
    "chunk_size": 512,
    "claim_extraction": "auto",
    "evidence_retrieval": "auto",
    "verification_model": "qwen3:14b",
    "top_k": 4,
    "output_dir": "outputs"
}
```

### Configuration Options

- **pdf_parser**: PDF parsing method (`"auto"`, `"docling_standard"`, `"nougat"`, `"pypdf2_fallback"`)
- **parser_kwargs**: Additional arguments for PDF parsing
- **chunk_size**: Maximum tokens per chunk (default: 512)
- **claim_extraction**: Claim extraction method (`"auto"`, `"fenice"`, `"rule_based"`)
- **evidence_retrieval**: Evidence retrieval method (`"auto"`, `"tfidf"`, `"bm25"`, `"sbert"`)
- **verification_model**: Ollama model for claim verification
- **top_k**: Number of evidence chunks per claim (default: 4)
- **output_dir**: Output directory (default: "outputs")

## Output Structure

The app creates the following directory structure:

```
outputs/
├── pdfs/                    # Downloaded PDFs
├── markdown/               # Parsed and cleaned markdown
├── chunks/                 # Chunked markdown for retrieval
├── reviews/                # Review files
│   ├── SUBMISSION_ID_all_reviews.json      # All reviews in one file
│   ├── SUBMISSION_ID_all_reviews.pkl       # Pickle backup of reviews
│   ├── SUBMISSION_ID_review_1_REVIEW_ID.json  # Individual review files
│   ├── SUBMISSION_ID_review_2_REVIEW_ID.json  # Individual review files
│   └── processed/          # Results from individual processing
│       ├── claims/         # Claims per review
│       ├── evidence/       # Evidence per review
│       └── verification/   # Verification per review
├── claims/                 # Extracted claims (all reviews combined)
├── evidence/               # Evidence retrieval results (all reviews)
├── verification/           # Claim verification results (all reviews)
└── *.json                  # Metadata and intermediate files
```

## Processing Pipeline

### Main Pipeline

1. **Crawl OpenReview**: Extract paper PDF and review data using improved API calls
2. **Parse PDF**: Convert PDF to markdown using selected parser
3. **Clean Markdown**: Remove artifacts and format content
4. **Chunk Text**: Split paper into manageable chunks
5. **Extract Reviews**: Structure review data with multiple fallback mechanisms
6. **Extract Claims**: Identify claims in review text
7. **Retrieve Evidence**: Find relevant paper sections for each claim
8. **Verify Claims**: Use LLM to verify claims against evidence

### Individual Review Processing

1. **Load Review**: Process individual review JSON files
2. **Extract Claims**: Extract claims from the specific review
3. **Retrieve Evidence**: Find relevant evidence for review claims
4. **Verify Claims**: Verify claims independently per review

### Review Extraction Improvements

- **Multiple API Methods**: Uses both `get_all_notes()` and `directReplies` fallback
- **Comprehensive Fields**: Extracts all available review fields (rating, confidence, strengths, weaknesses, etc.)
- **Multiple Formats**: Saves reviews as JSON, pickle, and individual files
- **Independent Processing**: Each review can be processed separately
- **Robust Fallbacks**: Multiple methods to load reviews if primary extraction fails

## Available Methods

### PDF Parsing
- **Docling Standard**: High-quality parsing with enrichment options
- **Nougat**: Meta's document understanding model
- **PyPDF2 Fallback**: Simple text extraction

### Claim Extraction
- **FENICE**: Neural claim extraction model
- **Rule-based**: Keyword-based extraction

### Evidence Retrieval
- **TF-IDF**: Traditional information retrieval
- **BM25**: Probabilistic retrieval model
- **SBERT**: Semantic similarity using sentence transformers

### Claim Verification
- **Ollama Models**: Any Ollama-compatible model (e.g., qwen3:14b, llama3.1:8b)

## Dependencies

The app gracefully handles missing dependencies by falling back to available methods. Core dependencies include:

- `openreview`: OpenReview API access
- `torch`: PyTorch for ML models
- `transformers`: Hugging Face transformers
- `scikit-learn`: For TF-IDF retrieval
- `rank-bm25`: For BM25 retrieval
- `sentence-transformers`: For SBERT retrieval
- `ollama`: For claim verification
- `tiktoken`: For text tokenization

## Troubleshooting

### Common Issues

1. **PDF Parsing Fails**: Try different parsing methods in config
2. **No Reviews Found**: 
   - Check if the OpenReview URL has published reviews
   - Try using OpenReview credentials for private reviews
   - Check individual review files in `outputs/reviews/` directory
3. **No Claims Extracted**: Check if reviews contain claim-like statements
4. **Evidence Retrieval Empty**: Verify paper chunks were created successfully
5. **Ollama Connection Error**: Ensure Ollama is running and model is available

### Review Processing Issues

1. **Reviews Not Extracted**: The app now uses multiple methods to fetch reviews:
   - Primary: `get_all_notes()` API call
   - Fallback: `directReplies` from submission object
   - Recovery: Load from saved JSON/pickle files
2. **Individual Review Processing**: Use `process_individual_review.py` to process reviews independently
3. **Missing Review Data**: Check the `*_all_reviews.json` and `*_all_reviews.pkl` files for raw review data

### Logs

The app creates detailed logs in `app.log` for debugging.

## Example Output

After processing, you'll get:
- Structured review data
- Extracted claims from reviews
- Evidence chunks for each claim
- Verification results with confidence scores
- Human-readable verification report

## Contributing

The app is modular and extensible. Each step is implemented as a separate module that can be easily modified or extended.
