#!/usr/bin/env python3
"""
Simple startup script for the GenAI Data Visualization Dashboard
"""

import uvicorn
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting GenAI Data Visualization Dashboard...")
    print("📍 API will be available at: http://localhost:8002")
    print("📚 Documentation at: http://localhost:8002/docs")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        uvicorn.run(
            "app.main_viz:app",
            host="127.0.0.1",
            port=8002,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server()