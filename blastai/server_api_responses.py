"""FastAPI endpoints for responses."""

import json
import time
import logging
import asyncio
from typing import List, Optional, Dict, Any, Union
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .response import AgentReasoning, AgentHistoryListResponse
from .engine import Engine

# Get logger for this module
logger = logging.getLogger(__name__)

class ResponseRequest(BaseModel):
    """Request format for /v1/responses endpoint."""
    model: str = Field(..., description="Model identifier")
    input: Union[str, List[Dict[str, Any]]] = Field(..., description="Input text or structured messages")
    previous_response_id: Optional[str] = Field(None, description="ID of previous response for stateful conversations")
    stream: bool = Field(False, description="Whether to stream responses")
    store: bool = Field(True, description="Whether to store response for future reference")
    instructions: Optional[str] = Field(None, description="System message for the conversation")
    cache_control: Optional[str] = Field("", description="Cache control settings (no-cache, no-store, no-cache-plan, no-store-plan)")

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

async def handle_responses(request: ResponseRequest, engine: Engine):
    """Handle responses requests."""
    # Extract task and check for previous response
    task = request.input if isinstance(request.input, str) else request.input[-1]["content"][0]["text"]
    has_previous = bool(request.previous_response_id)
    
    logger.debug(f"Received request to run task: {task}")
    
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

async def handle_delete_response(response_id: str, engine: Engine):
    """Delete a response and its associated task from cache.
    
    The response_id can be either:
    1. A task/response ID (e.g. "resp_123" or "chatcmpl-123")
    2. A task description
    """
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
    return JSONResponse({
        "id": response_id,
        "object": "response",
        "deleted": True
    })