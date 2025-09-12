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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
import uvicorn

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
    results: Optional[Dict[str, Any]] = Field(None, description="Processing results")

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
    submission_id = str(request.openreview_url).split("/")[-1]
    
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
        
        # Load configuration
        config = load_config()
        if request.config_overrides:
            config.update(request.config_overrides)
        
        # Create processor and run pipeline
        processor = OpenReviewProcessor(config)
        results = await asyncio.to_thread(
            processor.process_openreview_url,
            str(request.openreview_url)
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update job status
        processing_jobs[submission_id].update({
            "status": "completed",
            "end_time": datetime.now(),
            "processing_time": processing_time,
            "results": results
        })
        
        return ProcessPaperResponse(
            submission_id=submission_id,
            status="completed",
            message="Paper processed successfully",
            processing_time=processing_time,
            results=results
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
