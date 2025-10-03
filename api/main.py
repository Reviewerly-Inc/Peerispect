"""
FastAPI Application for Peerispect
REST API for OpenReview paper processing and claim verification
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl
import uvicorn
import hashlib

# Import our existing pipeline modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.main import OpenReviewProcessor

def load_config(config_path: str = "app/config.json") -> Dict[str, Any]:
    """Load configuration from JSON file with fallback to defaults"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            "pdf_parser": "auto",
            "parser_kwargs": {},
            "chunk_size": 512,
            "positional_chunking": True,
            "column_split_x": 300.0,
            "claim_extraction": "auto",
            "evidence_retrieval": "auto",
            "verification_model": "qwen3:8b",
            "top_k": 4,
            "output_dir": "outputs"
        }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Peerispect API",
    description="AI-powered OpenReview paper processing and claim verification system",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response schemas
class ProcessPaperRequest(BaseModel):
    """Request model for processing a paper"""
    openreview_url: HttpUrl = Field(..., description="OpenReview forum URL to process")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Optional configuration overrides")
    
class ProcessPaperResponse(BaseModel):
    """Response model for paper processing"""
    submission_id: str = Field(..., description="OpenReview submission ID")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="Verification results JSON")
    pdf_url: Optional[str] = Field(None, description="Web URL to access the PDF file")
    config_used: Optional[Dict[str, Any]] = Field(None, description="Configuration used for processing")
    cached: bool = Field(False, description="Whether result was served from cache")
    highlighting_map: Optional[Dict[str, Any]] = Field(None, description="Efficient mapping from claims to evidence positions for highlighting")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")

class ConfigResponse(BaseModel):
    """Configuration response"""
    config: Dict[str, Any] = Field(..., description="Current configuration")

# Global state for tracking processing jobs
processing_jobs: Dict[str, Dict[str, Any]] = {}

# Cache directory for API results
CACHE_DIR = Path("api_cache")
CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "pdfs").mkdir(exist_ok=True)
(CACHE_DIR / "results").mkdir(exist_ok=True)

def get_cache_key(url: str, config: Dict[str, Any]) -> str:
    """Generate cache key from URL and config"""
    config_str = json.dumps(config, sort_keys=True)
    combined = f"{url}|{config_str}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_submission_id_from_url(url: str) -> str:
    """Extract clean submission ID from OpenReview URL"""
    if "forum?id=" in url:
        return url.split("forum?id=")[-1]
    else:
        return url.split("/")[-1]

def load_positional_data(pipeline_results: Dict[str, Any], submission_id: str) -> Dict[str, Any]:
    """Load positional chunk data for highlighting evidence"""
    positional_data = {
        "chunks": {},
        "chunk_to_positions": {},
        "evidence_to_chunks": {}
    }
    
    try:
        # Check if positional chunking was used
        if pipeline_results.get("positional_dir"):
            positional_chunks_path = Path(pipeline_results["positional_dir"]) / "positional_chunks_v5.json"
            if positional_chunks_path.exists():
                with open(positional_chunks_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                # Create mapping from chunk ID to positions
                for chunk in chunks:
                    chunk_id = chunk['id']
                    positional_data["chunks"][chunk_id] = {
                        "text": chunk['text'],
                        "tokens": chunk['tokens'],
                        "pages": chunk['pages'],
                        "positions": chunk['positions'],
                        "spans_columns": chunk['spans_columns'],
                        "spans_pages": chunk['spans_pages']
                    }
                    positional_data["chunk_to_positions"][chunk_id] = chunk['positions']
                
                logger.info(f"Loaded {len(chunks)} positional chunks for highlighting")
            else:
                logger.warning(f"Positional chunks file not found: {positional_chunks_path}")
        else:
            logger.info("Positional chunking not used - no highlighting data available")
            
    except Exception as e:
        logger.error(f"Error loading positional data: {e}")
    
    return positional_data

def validate_cached_result(cached_data: Dict[str, Any]) -> bool:
    """
    Validate that cached result has all required data and is complete.
    
    Returns:
        bool: True if valid and complete, False otherwise
    """
    try:
        # Check required top-level fields
        required_fields = ["submission_id", "status", "results", "highlighting_map"]
        for field in required_fields:
            if field not in cached_data:
                logger.warning(f"Cached result missing required field: {field}")
                return False
        
        # Check status is completed
        if cached_data["status"] != "completed":
            logger.warning(f"Cached result status is not completed: {cached_data['status']}")
            return False
        
        # Check results is a non-empty list
        results = cached_data.get("results", [])
        if not isinstance(results, list) or len(results) == 0:
            logger.warning(f"Cached result has no verification results: {len(results) if isinstance(results, list) else 'not a list'}")
            return False
        
        # Check highlighting map structure
        highlighting_map = cached_data.get("highlighting_map", {})
        if not isinstance(highlighting_map, dict):
            logger.warning("Cached result highlighting_map is not a dict")
            return False
        
        evidence_registry = highlighting_map.get("evidence_registry", {})
        if not isinstance(evidence_registry, dict) or len(evidence_registry) == 0:
            logger.warning(f"Cached result has no evidence registry: {len(evidence_registry) if isinstance(evidence_registry, dict) else 'not a dict'}")
            return False
        
        # Check each result has required fields
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                logger.warning(f"Result {i} is not a dict")
                return False
            
            required_result_fields = ["claim_id", "claim", "evidence_ids", "verification"]
            for field in required_result_fields:
                if field not in result:
                    logger.warning(f"Result {i} missing required field: {field}")
                    return False
            
            # Check evidence_ids is a list
            evidence_ids = result.get("evidence_ids", [])
            if not isinstance(evidence_ids, list):
                logger.warning(f"Result {i} evidence_ids is not a list")
                return False
            
            # Check verification has required fields
            verification = result.get("verification", {})
            if not isinstance(verification, dict):
                logger.warning(f"Result {i} verification is not a dict")
                return False
            
            required_verification_fields = ["result", "confidence", "justification"]
            for field in required_verification_fields:
                if field not in verification:
                    logger.warning(f"Result {i} verification missing field: {field}")
                    return False
        
        # Check that all evidence_ids in results exist in evidence_registry
        all_evidence_ids = set()
        for result in results:
            all_evidence_ids.update(result.get("evidence_ids", []))
        
        for evidence_id in all_evidence_ids:
            if evidence_id not in evidence_registry:
                logger.warning(f"Evidence ID {evidence_id} referenced in results but not in evidence_registry")
                return False
        
        logger.info(f"Cached result validation passed: {len(results)} results, {len(evidence_registry)} evidence entries")
        return True
        
    except Exception as e:
        logger.error(f"Error validating cached result: {e}")
        return False

def create_highlighting_map(verification_data: List[Dict[str, Any]], positional_data: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Create efficient highlighting map with unique evidence registry and transform verification data"""
    highlighting_map = {
        "evidence_registry": {}  # evidence_id -> {text, positions, chunk_id}
    }
    transformed_verification_data = []
    
    try:
        logger.info(f"Processing {len(verification_data)} verification entries")
        logger.info(f"Positional data has {len(positional_data.get('chunks', {}))} chunks")
        
        if not verification_data or not positional_data.get("chunks"):
            logger.warning("No verification data or positional chunks available")
            return highlighting_map, verification_data
        
        evidence_text_to_id = {}  # text -> evidence_id
        evidence_id_counter = 1
        
        # Process all evidence and create unique registry
        for claim_data in verification_data:
            if isinstance(claim_data, str):
                continue
                
            claim_id = str(claim_data.get("claim_id", ""))
            evidence_list = claim_data.get("evidence", [])
            claim_evidence_ids = []
            
            for evidence in evidence_list:
                # Handle both string and dict evidence formats
                if isinstance(evidence, str):
                    evidence_text = evidence
                else:
                    evidence_text = evidence.get("text", "")
                
                if not evidence_text.strip():
                    continue
                
                # Check if we've already processed this evidence text
                if evidence_text in evidence_text_to_id:
                    evidence_id = evidence_text_to_id[evidence_text]
                    claim_evidence_ids.append(evidence_id)
                    continue
                
                # Check if evidence is already in the new format with chunk_id
                if isinstance(evidence, dict) and "chunk_id" in evidence:
                    # New format: evidence already has chunk_id
                    matching_chunk_id = evidence["chunk_id"]
                    best_match_score = 1.0
                else:
                    # Legacy format: find matching chunk using exact text matching
                    matching_chunk_id = None
                    best_match_score = 1.0  # Perfect match
                    
                    for chunk_id, chunk_data in positional_data["chunks"].items():
                        chunk_text = chunk_data["text"]
                        
                        # Exact text match (most robust)
                        if evidence_text.strip() == chunk_text.strip():
                            matching_chunk_id = chunk_id
                            best_match_score = 1.0
                            break
                        # Fallback: substring match for partial evidence
                        elif evidence_text.strip() in chunk_text.strip():
                            matching_chunk_id = chunk_id
                            best_match_score = 0.9
                            break
                
                if matching_chunk_id:
                    evidence_id = f"ev_{evidence_id_counter}"
                    evidence_id_counter += 1
                    
                    chunk_positions = positional_data["chunk_to_positions"].get(matching_chunk_id, [])
                    
                    # Store in registry
                    highlighting_map["evidence_registry"][evidence_id] = {
                        "text": evidence_text,
                        "chunk_id": matching_chunk_id,
                        "positions": chunk_positions,
                        "match_type": "exact" if best_match_score == 1.0 else "substring"
                    }
                    
                    evidence_text_to_id[evidence_text] = evidence_id
                    claim_evidence_ids.append(evidence_id)
                else:
                    # If no exact match found, create evidence without positional data
                    logger.warning(f"No matching chunk found for evidence: {evidence_text[:100]}...")
                    evidence_id = f"ev_{evidence_id_counter}"
                    evidence_id_counter += 1
                    
                    highlighting_map["evidence_registry"][evidence_id] = {
                        "text": evidence_text,
                        "chunk_id": None,
                        "positions": [],
                        "match_type": "no_match"
                    }
                    
                    evidence_text_to_id[evidence_text] = evidence_id
                    claim_evidence_ids.append(evidence_id)
            
            # Create transformed claim data with evidence IDs instead of text
            transformed_claim = {
                "claim_id": claim_data.get("claim_id"),
                "claim": claim_data.get("claim", ""),
                "evidence_ids": claim_evidence_ids,  # Replace evidence text with IDs
                "verification": claim_data.get("verification", {}),
                "review_id": claim_data.get("review_id", ""),
                "reviewer": claim_data.get("reviewer", "")
            }
            transformed_verification_data.append(transformed_claim)
        
        logger.info(f"Created efficient highlighting map: {len(highlighting_map['evidence_registry'])} unique evidence, {len(transformed_verification_data)} claims")
        
    except Exception as e:
        logger.error(f"Error creating highlighting map: {e}")
    
    return highlighting_map, transformed_verification_data

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Peerispect API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="2.0.0"
    )

@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration"""
    try:
        config = load_config()
        return ConfigResponse(config=config)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise HTTPException(status_code=500, detail="Failed to load configuration")

@app.get("/pdf/{submission_id}")
async def serve_pdf(submission_id: str):
    """Serve PDF file for a submission from cache"""
    pdf_path = CACHE_DIR / "pdfs" / f"{submission_id}.pdf"
    
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found in cache")
    
    return FileResponse(
        path=str(pdf_path),
        filename=f"{submission_id}.pdf",
        media_type="application/pdf"
    )

@app.post("/process", response_model=ProcessPaperResponse)
async def process_paper(request: ProcessPaperRequest, background_tasks: BackgroundTasks):
    """
    Process an OpenReview paper through the complete pipeline
    
    This endpoint:
    1. Crawls OpenReview for paper and reviews
    2. Downloads and parses the PDF
    3. Extracts and cleans review text
    4. Chunks the content
    5. Extracts claims from reviews
    6. Retrieves evidence from paper
    7. Verifies claims against evidence
    """
    submission_id = get_submission_id_from_url(str(request.openreview_url))
    
    # Load configuration
    config = load_config()
    if request.config_overrides:
        config.update(request.config_overrides)
    
    # Check cache first
    cache_key = get_cache_key(str(request.openreview_url), config)
    cache_file = CACHE_DIR / "results" / f"{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Validate cached result
            if validate_cached_result(cached_data):
                logger.info(f"Serving valid cached result for {submission_id}")
                return ProcessPaperResponse(
                    submission_id=cached_data["submission_id"],
                    status=cached_data["status"],
                    message=cached_data["message"],
                    processing_time=cached_data.get("processing_time"),
                    results=cached_data["results"],
                    pdf_url=cached_data.get("pdf_url"),
                    config_used=cached_data.get("config_used"),
                    cached=True,
                    highlighting_map=cached_data["highlighting_map"]
                )
            else:
                logger.warning(f"Cached result for {submission_id} failed validation, processing fresh")
                # Remove invalid cache file
                cache_file.unlink()
        except Exception as e:
            logger.error(f"Error loading cached result for {submission_id}: {e}")
            # Remove corrupted cache file
            if cache_file.exists():
                cache_file.unlink()
    
    # Check if already processing
    if submission_id in processing_jobs:
        return ProcessPaperResponse(
            submission_id=submission_id,
            status="already_processing",
            message="Paper is already being processed"
        )
    
    # Add to processing jobs
    processing_jobs[submission_id] = {
        "status": "processing",
        "start_time": datetime.now(),
        "url": str(request.openreview_url)
    }
    
    try:
        # Process the paper
        start_time = datetime.now()
        
        # Create processor and run pipeline
        processor = OpenReviewProcessor(config)
        pipeline_results = await asyncio.to_thread(
            processor.process_openreview_url,
            str(request.openreview_url)
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Debug pipeline results
        logger.info(f"Pipeline results keys: {list(pipeline_results.keys())}")
        logger.info(f"Verification path: {pipeline_results.get('verification_path')}")
        
        # Load verification results
        verification_data = []
        if pipeline_results.get("verification_path") and Path(pipeline_results["verification_path"]).exists():
            try:
                with open(pipeline_results["verification_path"], 'r') as f:
                    verification_data = json.load(f)
                    # Ensure it's a list if it's not already
                    if not isinstance(verification_data, list):
                        verification_data = [verification_data] if verification_data else []
                logger.info(f"Successfully loaded {len(verification_data)} verification entries")
            except Exception as e:
                logger.error(f"Error loading verification data: {e}")
                verification_data = []
        
        # Copy PDF to cache directory for serving
        original_pdf_path = Path(pipeline_results.get("pdf_path", ""))
        if original_pdf_path.exists():
            cached_pdf_path = CACHE_DIR / "pdfs" / f"{submission_id}.pdf"
            import shutil
            shutil.copy2(original_pdf_path, cached_pdf_path)
            logger.info(f"Copied PDF to cache: {cached_pdf_path}")
        
        # Load positional chunks for highlighting data
        positional_data = load_positional_data(pipeline_results, submission_id)
        
        # Create highlighting map and transform verification data
        logger.info(f"Creating highlighting map with {len(verification_data)} verification entries")
        try:
            highlighting_map, transformed_verification_data = create_highlighting_map(verification_data, positional_data)
            logger.info(f"Created highlighting map with {len(highlighting_map.get('evidence_registry', {}))} evidence entries")
        except Exception as e:
            logger.error(f"Error creating highlighting map: {e}")
            highlighting_map = {"evidence_registry": {}}
            transformed_verification_data = verification_data
        
        # Update job status
        processing_jobs[submission_id].update({
            "status": "completed",
            "end_time": datetime.now(),
            "processing_time": processing_time,
            "results": pipeline_results
        })
        
        # Prepare response data
        response_data = {
            "submission_id": submission_id,
            "status": "completed",
            "message": "Paper processed successfully",
            "processing_time": processing_time,
            "results": transformed_verification_data,
            "pdf_url": f"/pdf/{submission_id}",
            "config_used": config,
            "cached": False,
            "highlighting_map": highlighting_map
        }
        
        # Cache the result if validation passes
        try:
            # Validate the response before caching
            if validate_cached_result(response_data):
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Cached successful result for {submission_id}")
            else:
                logger.warning(f"Result validation failed, not caching {submission_id}")
        except Exception as e:
            logger.error(f"Error caching result for {submission_id}: {e}")
        
        return ProcessPaperResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing paper {submission_id}: {e}")
        
        # Update job status
        processing_jobs[submission_id].update({
            "status": "failed",
            "end_time": datetime.now(),
            "error": str(e)
        })
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process paper: {str(e)}"
        )

@app.get("/status/{submission_id}", response_model=Dict[str, Any])
async def get_processing_status(submission_id: str):
    """Get processing status for a specific submission"""
    if submission_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    job = processing_jobs[submission_id]
    return {
        "submission_id": submission_id,
        "status": job["status"],
        "start_time": job["start_time"],
        "end_time": job.get("end_time"),
        "processing_time": job.get("processing_time"),
        "error": job.get("error")
    }

@app.get("/jobs", response_model=Dict[str, List[Dict[str, Any]]])
async def list_processing_jobs():
    """List all processing jobs"""
    return {
        "jobs": [
            {
                "submission_id": sid,
                "status": job["status"],
                "start_time": job["start_time"],
                "url": job["url"]
            }
            for sid, job in processing_jobs.items()
        ]
    }

@app.delete("/jobs/{submission_id}")
async def clear_processing_job(submission_id: str):
    """Clear a specific processing job from memory"""
    if submission_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    del processing_jobs[submission_id]
    return {"message": f"Job {submission_id} cleared"}

@app.delete("/jobs")
async def clear_all_jobs():
    """Clear all processing jobs from memory"""
    processing_jobs.clear()
    return {"message": "All jobs cleared"}

@app.get("/cache")
async def get_cache_info():
    """Get cache information"""
    results_dir = CACHE_DIR / "results"
    pdfs_dir = CACHE_DIR / "pdfs"
    
    cached_files = list(results_dir.glob("*.json")) if results_dir.exists() else []
    cached_pdfs = list(pdfs_dir.glob("*.pdf")) if pdfs_dir.exists() else []
    
    cached_entries = []
    for cache_file in cached_files:
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                cached_entries.append({
                    "submission_id": data.get("submission_id"),
                    "cached_at": data.get("cached_at"),
                    "config": data.get("config_used"),
                    "cache_file": str(cache_file)
                })
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {e}")
    
    return {
        "cache_size": len(cached_files),
        "pdf_count": len(cached_pdfs),
        "cached_entries": cached_entries
    }

@app.delete("/cache")
async def clear_cache():
    """Clear all cached results and PDFs"""
    import shutil
    
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / "pdfs").mkdir(exist_ok=True)
        (CACHE_DIR / "results").mkdir(exist_ok=True)
    
    return {"message": "Cache cleared"}

@app.delete("/cache/{submission_id}")
async def clear_cache_entry(submission_id: str):
    """Clear cache entry for a specific submission"""
    removed_count = 0
    
    # Remove PDF
    pdf_path = CACHE_DIR / "pdfs" / f"{submission_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
        removed_count += 1
    
    # Remove result files for this submission
    results_dir = CACHE_DIR / "results"
    if results_dir.exists():
        for cache_file in results_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if data.get("submission_id") == submission_id:
                        cache_file.unlink()
                        removed_count += 1
            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {e}")
    
    return {"message": f"Cache entries for {submission_id} cleared", "removed_count": removed_count}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5015,
        reload=True,
        log_level="info"
    )
