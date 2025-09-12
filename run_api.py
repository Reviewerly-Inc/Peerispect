#!/usr/bin/env python3
"""
Startup script for Peeriscope.V2 FastAPI server
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

if __name__ == "__main__":
    import uvicorn
    from api.main import app
    
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting Peeriscope.V2 API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Alternative docs: http://localhost:8000/redoc")
    print("❤️  Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
