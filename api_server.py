"""
Simplified entry point — launches the FastAPI server.
All logic has been moved to the app/ package.

Usage:
    python api_server.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
