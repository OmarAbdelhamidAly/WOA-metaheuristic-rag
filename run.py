"""
Production and development entry point.
Reads $PORT dynamically from the environment (default: 5000) for Render compatibility.
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"============================================================")
    print(f" Starting Whale Optimization & RAG Simulator on {host}:{port}")
    print(f" Local URL: http://localhost:{port}")
    print(f"============================================================")
    uvicorn.run("server:app", host=host, port=port, log_level="info")
