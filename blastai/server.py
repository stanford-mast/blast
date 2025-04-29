"""FastAPI server providing OpenAI-compatible API endpoints."""

# Set anonymized telemetry to false before any imports
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"

import sys
import json
import yaml
import time
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional, Union, Dict, Any, TYPE_CHECKING
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Settings, Constraints
from .logging_setup import setup_logging

# Get logger for this module
logger = logging.getLogger(__name__)

# Global state
_engine: Optional["Engine"] = None
_settings: Optional[Settings] = None
_constraints: Optional[Constraints] = None

# Initialize logging with default settings
setup_logging(Settings())

# Only import these after logging is configured
from .engine import Engine
from .response import AgentReasoning, AgentHistoryListResponse

def load_config(config_path: Optional[str] = None) -> tuple[Settings, Constraints]:
    """Load configuration from file or use defaults.
    
    Args:
        config_path: Optional path to config YAML file
        
    Returns:
        Tuple of (Settings, Constraints)
    """
    # Load default config
    default_config_path = Path(__file__).parent / 'default_config.yaml'
    with open(default_config_path) as f:
        config = yaml.safe_load(f)
        
    # Override with user config if provided
    if config_path:
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
            # Update nested dicts
            if 'settings' in user_config:
                config['settings'].update(user_config['settings'])
            if 'constraints' in user_config:
                config['constraints'].update(user_config['constraints'])
    
    # Create Settings object and configure logging immediately
    settings = Settings(
        persist_cache=config['settings']['persist_cache'],
        browser_use_log_level=config['settings']['browser_use_log_level'],
        blastai_log_level=config['settings']['blastai_log_level']
    )
    
    # Configure logging with new settings
    setup_logging(settings)
    
    constraints = Constraints.create(
        max_memory=config['constraints']['max_memory'],
        max_concurrent_browsers=config['constraints']['max_concurrent_browsers'],
        allow_parallelism=config['constraints']['allow_parallelism'],
        llm_model=config['constraints']['llm_model'],
        allow_vision=config['constraints']['allow_vision'],
        require_headless=config['constraints']['require_headless'],
        share_browser_process=config['constraints']['share_browser_process']
    )
    
    return settings, constraints

def init_app_state(config_path: Optional[str] = None):
    """Initialize global app state with config.
    
    Args:
        config_path: Optional path to config YAML file
    """
    global _settings, _constraints
    _settings, _constraints = load_config(config_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    global _engine
    shutdown_event = asyncio.Event()
    
    # Define shutdown handler first
    async def handle_shutdown():
        if _engine:
            try:
                await asyncio.wait_for(_engine.stop(), timeout=30.0)
            except Exception as e:
                logger.error(f"Error stopping engine: {e}")
        shutdown_event.set()
    
    try:
        # Initialize engine
        if not _engine:
            if not _settings or not _constraints:
                # Initialize with defaults if not already initialized
                init_app_state()
            _engine = Engine(settings=_settings, constraints=_constraints)
            await _engine.start()
        
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
    version="0.1.0",
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

# Add middleware to log requests
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests."""
    response = await call_next(request)
    return response

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
        raise HTTPException(status_code=500, detail=str(e))


class Message(BaseModel):
    """OpenAI-style chat message."""
    role: str = Field(..., description="Role of the message sender (system/user/assistant)")
    content: str = Field(..., description="Content of the message")
    cache_control: Optional[str] = Field("", description="Cache control settings (no-cache, no-store, no-cache-plan, no-store-plan)")

class ChatCompletionRequest(BaseModel):
    """Request format for /v1/chat/completions endpoint."""
    model: str = Field(..., description="Model identifier")
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    stream: bool = Field(False, description="Whether to stream responses")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")

class ResponseRequest(BaseModel):
    """Request format for /v1/responses endpoint."""
    model: str = Field(..., description="Model identifier")
    input: Union[str, List[Dict[str, Any]]] = Field(..., description="Input text or structured messages")
    previous_response_id: Optional[str] = Field(None, description="ID of previous response for stateful conversations")
    stream: bool = Field(False, description="Whether to stream responses")
    store: bool = Field(True, description="Whether to store response for future reference")
    instructions: Optional[str] = Field(None, description="System message for the conversation")
    cache_control: Optional[str] = Field("", description="Cache control settings (no-cache, no-store, no-cache-plan, no-store-plan)")

async def get_engine() -> Engine:
    """Get the global engine instance, creating it if needed."""
    global _engine
    if _engine is None:
        if not _settings or not _constraints:
            # Initialize with defaults if not already initialized
            init_app_state()
        _engine = Engine(settings=_settings, constraints=_constraints)
        await _engine.start()  # Wait for engine to start
        
    return _engine

async def format_chat_stream(engine_stream, model: str):
    """Format chat completions streaming response."""
    created = int(time.time())
    known_tasks = set()
    
    async for update in engine_stream:
        if isinstance(update, AgentReasoning):
            task_id = update.task_id
            response_id = f"chatcmpl-{task_id}"  # Keep task ID visible
            system_fingerprint = f"fp_{task_id[-8:]}"
            
            # Send initial role chunk for new tasks
            if task_id not in known_tasks:
                known_tasks.add(task_id)
                data = {
                    'id': response_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': model,
                    'system_fingerprint': system_fingerprint,
                    'service_tier': 'default',
                    'usage': None,
                    'choices': [{
                        'index': 0,
                        'delta': {
                            'role': 'assistant',
                            'content': "",
                            'function_call': None,
                            'refusal': None,
                            'tool_calls': None
                        },
                        'finish_reason': None,
                        'logprobs': None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            # Send content chunk based on type
            # No prefix or indicator of type for now as we can interpret
            # based on the last one being the final result and anything before
            # is either a screenshot or thought.
            data = {
                'id': response_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model,
                'system_fingerprint': system_fingerprint,
                'service_tier': 'default',
                'usage': None,
                'choices': [{
                    'index': 0,
                    'delta': {
                        'content': update.content,
                        'function_call': None,
                        'refusal': None,
                        'role': None,
                        'tool_calls': None
                    },
                    'finish_reason': None,
                    'logprobs': None
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
                
        elif isinstance(update, AgentHistoryListResponse):
            task_id = update.task_id
            response_id = f"chatcmpl-{task_id}"
            system_fingerprint = f"fp_{task_id[-8:]}"
            
            # Send final result chunk
            data = {
                'id': response_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model,
                'system_fingerprint': system_fingerprint,
                'service_tier': 'default',
                'usage': None,
                'choices': [{
                    'index': 0,
                    'delta': {
                        'content': update.final_result(),
                        'function_call': None,
                        'refusal': None,
                        'role': None,
                        'tool_calls': None
                    },
                    'finish_reason': None,
                    'logprobs': None
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            # Send completion chunk
            data = {
                'id': response_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model,
                'system_fingerprint': system_fingerprint,
                'service_tier': 'default',
                'usage': None,
                'choices': [{
                    'index': 0,
                    'delta': {
                        'content': None,
                        'function_call': None,
                        'refusal': None,
                        'role': None,
                        'tool_calls': None
                    },
                    'finish_reason': 'stop',
                    'logprobs': None
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"

async def format_response_stream(engine_stream, model: str, request: ResponseRequest):
    """Format responses streaming response."""
    created_at = int(time.time())
    
    # Track output items and content parts
    output_items = {}  # task_id -> output item
    content_parts = {}  # task_id -> list of content parts
    response_id = None
    
    # Process stream updates
    async for update in engine_stream:
        # Get task_id and set response_id on first update
        task_id = None
        if isinstance(update, AgentHistoryListResponse):
            task_id = update.task_id
        elif isinstance(update, AgentReasoning):
            task_id = update.task_id
            
        if task_id and not response_id:
            response_id = f"resp_{task_id}"
            # Initial response created
            initial_response = {
                'id': response_id,
                'object': 'response',
                'created_at': created_at,
                'status': 'in_progress',
                'error': None,
                'incomplete_details': None,
                'instructions': request.instructions,
                'max_output_tokens': None,
                'model': model,
                'output': [],
                'parallel_tool_calls': True,
                'previous_response_id': request.previous_response_id,
                'reasoning': {
                    'effort': None,
                    'generate_summary': None,
                    'summary': None
                },
                'store': request.store,
                'temperature': 1.0,
                'text': {'format': {'type': 'text'}},
                'tool_choice': 'auto',
                'tools': [],
                'top_p': 1.0,
                'truncation': 'disabled',
                'usage': None,
                'user': None,
                'metadata': {},
                'service_tier': 'default'  # Always use default for consistency
            }
            data = {
                'type': 'response.created',
                'response': initial_response
            }
            yield f"event: response.created\ndata: {json.dumps(data)}\n\n"
            
            # Response in progress
            data = {
                'type': 'response.in_progress',
                'response': initial_response
            }
            yield f"event: response.in_progress\ndata: {json.dumps(data)}\n\n"
        
        if isinstance(update, AgentReasoning):
            task_id = update.task_id
            msg_id = f"msg_{task_id}"  # Keep task ID visible
            
            # Add output item if new task
            if task_id not in output_items:
                # Create output item
                output_items[task_id] = {
                    'id': msg_id,
                    'type': 'message',
                    'status': 'in_progress',
                    'role': 'assistant',
                    'content': []
                }
                content_parts[task_id] = []
                
                # Send output item added event
                data = {
                    'type': 'response.output_item.added',
                    'output_index': len(output_items) - 1,
                    'item': output_items[task_id]
                }
                yield f"event: response.output_item.added\ndata: {json.dumps(data)}\n\n"
            
            # Add content part based on type
            content_index = len(content_parts[task_id])
            part = {
                'type': 'output_text',
                'text': update.content,
                'annotations': []
            }
            content_parts[task_id].append(part)
            
            # Content part added with empty text
            data = {
                'type': 'response.content_part.added',
                'item_id': msg_id,
                'output_index': len(output_items) - 1,
                'content_index': content_index,
                'part': {
                    'type': 'output_text',
                    'text': "",
                    'annotations': []
                }
            }
            yield f"event: response.content_part.added\ndata: {json.dumps(data)}\n\n"
            
            # Content delta
            data = {
                'type': 'response.output_text.delta',
                'item_id': msg_id,
                'output_index': len(output_items) - 1,
                'content_index': content_index,
                'delta': update.content
            }
            yield f"event: response.output_text.delta\ndata: {json.dumps(data)}\n\n"
            
            # Content done
            data = {
                'type': 'response.output_text.done',
                'item_id': msg_id,
                'output_index': len(output_items) - 1,
                'content_index': content_index,
                'text': update.content
            }
            yield f"event: response.output_text.done\ndata: {json.dumps(data)}\n\n"
            
            # Content part done
            data = {
                'type': 'response.content_part.done',
                'item_id': msg_id,
                'output_index': len(output_items) - 1,
                'content_index': content_index,
                'part': part
            }
            yield f"event: response.content_part.done\ndata: {json.dumps(data)}\n\n"
            
        elif isinstance(update, AgentHistoryListResponse):
            task_id = update.task_id
            msg_id = f"msg_{task_id}"  # Keep task ID visible
            
            # Initialize content_parts for task if needed (for cached results)
            if task_id not in content_parts:
                content_parts[task_id] = []
                output_items[task_id] = {
                    'id': msg_id,
                    'type': 'message',
                    'status': 'in_progress',
                    'role': 'assistant',
                    'content': []
                }
                # Send output item added event
                data = {
                    'type': 'response.output_item.added',
                    'output_index': len(output_items) - 1,
                    'item': output_items[task_id]
                }
                yield f"event: response.output_item.added\ndata: {json.dumps(data)}\n\n"
            
            # Add final result content part
            content_index = len(content_parts[task_id])
            part = {
                'type': 'output_text',
                'text': update.final_result(),
                'annotations': []
            }
            content_parts[task_id].append(part)
            
            # Content part added with empty text
            data = {
                'type': 'response.content_part.added',
                'item_id': msg_id,
                'output_index': len(output_items) - 1,
                'content_index': content_index,
                'part': {
                    'type': 'output_text',
                    'text': "",
                    'annotations': []
                }
            }
            yield f"event: response.content_part.added\ndata: {json.dumps(data)}\n\n"
            
            # Content delta
            data = {
                'type': 'response.output_text.delta',
                'item_id': msg_id,
                'output_index': len(output_items) - 1,
                'content_index': content_index,
                'delta': update.final_result()
            }
            yield f"event: response.output_text.delta\ndata: {json.dumps(data)}\n\n"
            
            # Content done
            data = {
                'type': 'response.output_text.done',
                'item_id': msg_id,
                'output_index': len(output_items) - 1,
                'content_index': content_index,
                'text': update.final_result()
            }
            yield f"event: response.output_text.done\ndata: {json.dumps(data)}\n\n"
            
            # Content part done
            data = {
                'type': 'response.content_part.done',
                'item_id': msg_id,
                'output_index': len(output_items) - 1,
                'content_index': content_index,
                'part': part
            }
            yield f"event: response.content_part.done\ndata: {json.dumps(data)}\n\n"
            
            # Mark output item as completed
            output_items[task_id]['status'] = 'completed'
            output_items[task_id]['content'] = content_parts[task_id]
            data = {
                'type': 'response.output_item.done',
                'output_index': len(output_items) - 1,
                'item': output_items[task_id]
            }
            yield f"event: response.output_item.done\ndata: {json.dumps(data)}\n\n"
            
            # Response completed with detailed usage stats
            final_response = {
                'id': response_id,
                'object': 'response',
                'created_at': created_at,
                'status': 'completed',
                'error': None,
                'incomplete_details': None,
                'instructions': request.instructions,
                'max_output_tokens': None,
                'model': model,
                'output': list(output_items.values()),
                'parallel_tool_calls': True,
                'previous_response_id': request.previous_response_id,
                'reasoning': {
                    'effort': None,
                    'generate_summary': None,
                    'summary': None
                },
                'store': request.store,
                'temperature': 1.0,
                'text': {'format': {'type': 'text'}},
                'tool_choice': 'auto',
                'tools': [],
                'top_p': 1.0,
                'truncation': 'disabled',
                'usage': {
                    'input_tokens': 0,  # TODO: Get actual counts
                    'input_tokens_details': {
                        'cached_tokens': 0
                    },
                    'output_tokens': 0,
                    'output_tokens_details': {
                        'reasoning_tokens': 0
                    },
                    'total_tokens': 0
                },
                'user': None,
                'metadata': {},
                'service_tier': 'default'  # Always use default for consistency
            }
            data = {
                'type': 'response.completed',
                'response': final_response
            }
            yield f"event: response.completed\ndata: {json.dumps(data)}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completions requests."""
    # Extract last task and check for conversation history
    last_task = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
    logger.debug(f"Received request to run task: {last_task}")
    
    try:
        # Get global engine
        engine = await get_engine()
        
        # Extract tasks and cache controls
        tasks = []
        cache_controls = []
        for msg in request.messages:
            if msg.role == "user":
                tasks.append(msg.content)
                cache_controls.append(msg.cache_control or "")
        
        if request.stream:
            try:
                result = await engine.run(tasks, cache_control=cache_controls, stream=True)
                return StreamingResponse(
                    format_chat_stream(result, request.model),
                    media_type="text/event-stream"
                )
            except asyncio.TimeoutError:
                logger.error("Task failed in stream_task_events: timeout")
                raise HTTPException(status_code=504, detail="Request timed out")
            except Exception as e:
                logger.error(f"Task failed in stream_task_events: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            try:
                result = await engine.run(tasks, cache_control=cache_controls)
                return JSONResponse({
                    "id": f"chatcmpl-{result.response_id.split('_')[1]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.final_result(),
                        },
                        "finish_reason": "stop"
                    }]
                })
            except asyncio.TimeoutError:
                logger.error("Task failed in get_task_result: timeout")
                raise HTTPException(status_code=504, detail="Request timed out")
            except Exception as e:
                logger.error(f"Task failed in get_task_result: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Task failed: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/responses")
async def responses(request: ResponseRequest):
    """Handle responses requests."""
    # Extract task and check for previous response
    task = request.input if isinstance(request.input, str) else request.input[-1]["content"][0]["text"]
    has_previous = bool(request.previous_response_id)
    
    logger.debug(f"Received request to run task: {task}")
    
    engine = await get_engine()
    
    # Handle input
    if isinstance(request.input, str):
        tasks = [request.input]
    else:
        tasks = [msg["content"][0]["text"] for msg in request.input
                if msg["role"] == "user" and msg["content"][0]["type"] == "input_text"]
    
    if request.stream:
        result = await engine.run(
            tasks,
            stream=True,
            previous_response_id=request.previous_response_id,
            cache_control=request.cache_control
        )
        return StreamingResponse(
            format_response_stream(result, request.model, request),
            media_type="text/event-stream"
        )
    else:
        result = await engine.run(
            tasks,
            previous_response_id=request.previous_response_id,
            cache_control=request.cache_control
        )
        
        # Get task ID and final result
        task_id = result.task_id
        final_result = result.final_result()
        
        # Construct response matching OpenAI format
        response = {
            "id": f"resp_{task_id}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": request.instructions,
            "max_output_tokens": None,
            "model": request.model,
            "output": [{
                "type": "message",
                "id": f"msg_{task_id}",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": final_result,
                    "annotations": []
                }]
            }],
            "parallel_tool_calls": True,
            "previous_response_id": request.previous_response_id,
            "reasoning": {
                "effort": None,
                "generate_summary": None,
                "summary": None
            },
            "store": request.store,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 0,  # TODO: Get actual counts
                "input_tokens_details": {
                    "cached_tokens": 0
                },
                "output_tokens": 0,
                "output_tokens_details": {
                    "reasoning_tokens": 0
                },
                "total_tokens": 0
            },
            "user": None,
            "metadata": {},
            "service_tier": "default"
        }
        return JSONResponse(response)

@app.delete("/responses/{response_id}")
async def delete_response(response_id: str):
    """Delete a response and its associated task from cache.
    
    The response_id can be either:
    1. A task/response ID (e.g. "resp_123" or "chatcmpl-123")
    2. A task description
    """
    engine = await get_engine()
    
    # First try to interpret as task ID
    task_id = None
    if response_id.startswith("resp_"):
        task_id = response_id.split('_')[1]
    elif response_id.startswith("chatcmpl-"):
        task_id = response_id.split('-')[1]
        
    if task_id:
        # Try to get task by ID
        task = engine.scheduler.tasks.get(task_id)
        if task:
            # Remove from cache using task's lineage
            engine.cache_manager.remove_task(task.lineage)
            return JSONResponse({
                "id": response_id,
                "object": "response",
                "deleted": True
            })
    
    # If not found by ID, treat as description
    engine.cache_manager.remove_task([response_id])
    # Same id as what was passed in whether it be a task ID or description
    return JSONResponse({
        "id": response_id,
        "object": "response",
        "deleted": True
    })