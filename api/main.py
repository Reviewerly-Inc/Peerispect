"""
FastAPI Application for Peeriscope.V2
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
sys.path.append(str(Path(__file__).parent.parent / "app"))

from main import OpenReviewProcessor

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
    title="Peeriscope.V2 API",
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

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Peeriscope.V2 API",
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
    
    # Check cache first - look for existing cached results
    cache_key = get_cache_key(str(request.openreview_url), config)
    cache_file = CACHE_DIR / "results" / f"{cache_key}.json"
    
    if cache_file.exists():
        logger.info(f"Serving cached result for {submission_id}")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        
        verification_data = cached_data.get("verification_data", [])
        # Ensure it's a list if it's not already
        if not isinstance(verification_data, list):
            verification_data = [verification_data] if verification_data else []
            
        return ProcessPaperResponse(
            submission_id=submission_id,
            status="completed",
            message="Paper processed successfully (cached)",
            processing_time=0.0,  # Cached results have no processing time
            results=verification_data,
            pdf_url=f"/pdf/{submission_id}",
            config_used=config,
            cached=True
        )
    
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
        
        # Load verification results
        verification_data = []
        if pipeline_results.get("verification_path") and Path(pipeline_results["verification_path"]).exists():
            with open(pipeline_results["verification_path"], 'r') as f:
                verification_data = json.load(f)
                # Ensure it's a list if it's not already
                if not isinstance(verification_data, list):
                    verification_data = [verification_data] if verification_data else []
        
        # Copy PDF to cache directory
        original_pdf_path = Path(pipeline_results.get("pdf_path", ""))
        if original_pdf_path.exists():
            cached_pdf_path = CACHE_DIR / "pdfs" / f"{submission_id}.pdf"
            import shutil
            shutil.copy2(original_pdf_path, cached_pdf_path)
            logger.info(f"Copied PDF to cache: {cached_pdf_path}")
        
        # Save results to cache
        cache_data = {
            "verification_data": verification_data,
            "submission_id": submission_id,
            "config_used": config,
            "cached_at": datetime.now().isoformat(),
            "processing_time": processing_time,
            "original_results": pipeline_results
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Cached results for {submission_id} at {cache_file}")
        
        # Update job status
        processing_jobs[submission_id].update({
            "status": "completed",
            "end_time": datetime.now(),
            "processing_time": processing_time,
            "results": pipeline_results
        })
        
        return ProcessPaperResponse(
            submission_id=submission_id,
            status="completed",
            message="Paper processed successfully",
            processing_time=processing_time,
            results=verification_data,
            pdf_url=f"/pdf/{submission_id}",
            config_used=config,
            cached=False
        )
        
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
        port=8000,
        reload=True,
        log_level="info"
    )
