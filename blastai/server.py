"""FastAPI server providing OpenAI-compatible API endpoints."""

# Set anonymized telemetry to false before any imports
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .server_api_chat_completions import ChatCompletionRequest, handle_chat_completions
from .server_api_responses import ResponseRequest, handle_responses, handle_delete_response
from .server_api_realtime import (
    RealtimeConnection,
    handle_realtime_connection,
    cleanup_stale_connections,
    TaskRequest,
    RealtimeMessage,
    MessageType
)
from .engine import Engine

# Get logger for this module
logger = logging.getLogger(__name__)

# Global engine instance
_engine: Optional[Engine] = None

# Active WebSocket connections
_websocket_connections: Dict[str, RealtimeConnection] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    global _engine
    shutdown_event = asyncio.Event()
    cleanup_task = None
    
    # Define shutdown handler first
    async def handle_shutdown():
        if _engine:
            try:
                await asyncio.wait_for(_engine.stop(), timeout=30.0)
            except Exception as e:
                logger.error(f"Error stopping engine: {e}")
        
        # Cancel cleanup task if running
        if cleanup_task and not cleanup_task.done():
            logger.info("Cancelling connection cleanup task")
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
            
        shutdown_event.set()
    
    try:
        # Initialize engine if needed
        if not _engine:
            _engine = await Engine.create()
            logger.info("Engine started successfully")
        
        # Start connection cleanup task
        cleanup_task = asyncio.create_task(cleanup_stale_connections(_websocket_connections))
        logger.info("Started connection cleanup task")
        
        # Store shutdown handler
        app.state.handle_shutdown = handle_shutdown
        
        try:
            yield
        except asyncio.CancelledError:
            # Handle cancellation explicitly
            await handle_shutdown()
            # Wait briefly for cleanup to complete
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            raise
    finally:
        if not shutdown_event.is_set():
            await handle_shutdown()

app = FastAPI(
    title="BlastAI API",
    version="0.1.6",
    lifespan=lifespan
)

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins since we're running locally
    allow_credentials=False,  # Don't need credentials
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
    max_age=1  # Short cache for development
)

# Add middleware to log requests and errors
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests and errors."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error handling request: {str(e)}", exc_info=True)
        raise

@app.get("/ping")
async def ping():
    """Simple ping endpoint for testing."""
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        ready = _engine is not None and _engine.scheduler is not None
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok" if ready else "initializing",
                "ready": ready
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "ready": False
            }
        )

@app.get("/metrics")
async def get_metrics():
    """Get current server metrics."""
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
        
    try:
        # Get metrics from engine
        metrics = await _engine.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        # Re-raise as HTTPException but ensure error is logged first
        raise HTTPException(status_code=500, detail=str(e)) from e

async def get_engine() -> Engine:
    """Get the global engine instance, creating it if needed."""
    global _engine
    if _engine is None:
        _engine = await Engine.create()
        logger.info("Engine started successfully")
    return _engine

# Chat completions endpoint
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completions requests."""
    engine = await get_engine()
    return await handle_chat_completions(request, engine)

# Responses endpoints
@app.post("/responses")
async def responses(request: ResponseRequest):
    """Handle responses requests."""
    engine = await get_engine()
    return await handle_responses(request, engine)

@app.delete("/responses/{response_id}")
async def delete_response(response_id: str):
    """Delete a response and its associated task from cache."""
    engine = await get_engine()
    return await handle_delete_response(response_id, engine)

@app.websocket("/realtime")
async def realtime_endpoint(websocket: WebSocket):
    """WebSocket endpoint for realtime task execution.
    
    Enables:
    - Interactive task execution
    - Task stopping
    - Human-in-the-loop communication
    - Task chaining with prerequisites
    """
    engine = await get_engine()
    await handle_realtime_connection(websocket, engine, _websocket_connections)