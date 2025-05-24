"""FastAPI endpoints for chat completions."""

import json
import time
import logging
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .response import AgentReasoning, AgentHistoryListResponse
from .engine import Engine

# Get logger for this module
logger = logging.getLogger(__name__)

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
            
            # Send content chunk
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

async def handle_chat_completions(request: ChatCompletionRequest, engine: Engine):
    """Handle chat completions requests."""
    # Extract last task and check for conversation history
    last_task = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
    logger.debug(f"Received request to run task: {last_task}")
    
    try:
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