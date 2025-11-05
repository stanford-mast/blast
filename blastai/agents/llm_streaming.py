"""
Streaming LLM call wrapper with Time-To-First-Token (TTFT) measurement.

Provides utilities to call LLMs with streaming enabled, measuring:
- Time to first token (network + model startup latency)
- Token generation speed (tokens per second)
- Total tokens generated

This is useful for understanding LLM performance bottlenecks:
- High TTFT → network/routing issues
- Low tokens/sec → model capacity issues
"""

import time
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer

logger = logging.getLogger(__name__)


@dataclass
class StreamingTiming:
    """Detailed timing information from a streaming LLM call."""
    total_seconds: float
    time_to_first_token: Optional[float] = None  # Latency until first token arrives
    generation_seconds: Optional[float] = None    # Time spent generating (total - TTFT)
    tokens_per_second: Optional[float] = None     # Average generation speed
    total_tokens: Optional[int] = None            # Total tokens in completion
    total_characters: int = 0                     # Total characters (for rough token estimate)


async def stream_llm_call(
    llm: BaseChatModel,
    messages: list[BaseMessage]
) -> Tuple[str, StreamingTiming]:
    """
    Call LLM with streaming enabled and measure detailed timing.
    
    If the LLM doesn't support streaming (or we can't access the client),
    falls back to regular ainvoke() with limited timing info.
    
    Args:
        llm: LLM instance (must have get_client() method for streaming)
        messages: List of conversation messages
        
    Returns:
        Tuple of (completion_text, timing_info)
        
    Example:
        ```python
        llm = ChatOpenAI(model="gpt-4o")
        messages = [UserMessage(content="Hello")]
        
        completion, timing = await stream_llm_call(llm, messages)
        
        print(f"TTFT: {timing.time_to_first_token:.2f}s")
        print(f"Generation: {timing.tokens_per_second:.1f} tokens/sec")
        ```
    """
    start_time = time.time()
    
    # Check provider and use appropriate streaming implementation
    provider = getattr(llm, 'provider', None)
    
    # Try streaming if LLM supports it
    if hasattr(llm, 'get_client') and hasattr(llm, 'model'):
        try:
            if provider == 'groq':
                # Use Groq-specific streaming
                return await _stream_groq(llm, messages, start_time)
            else:
                # Default to OpenAI-compatible streaming
                return await _stream_openai(llm, messages, start_time)
        except Exception as e:
            logger.warning(f"Streaming failed, falling back to non-streaming: {e}")
            # Fall through to non-streaming
    
    # Fallback to non-streaming
    response = await llm.ainvoke(messages)
    total_time = time.time() - start_time
    
    # Extract token count if available
    total_tokens = None
    if hasattr(response, 'usage') and response.usage:
        if hasattr(response.usage, 'completion_tokens'):
            total_tokens = response.usage.completion_tokens
    
    timing = StreamingTiming(
        total_seconds=total_time,
        time_to_first_token=None,  # Unknown in non-streaming mode
        generation_seconds=None,
        tokens_per_second=total_tokens / total_time if total_tokens and total_time > 0 else None,
        total_tokens=total_tokens,
        total_characters=len(response.completion)
    )
    
    return response.completion, timing


async def _stream_openai_compatible(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    start_time: float,
    serializer_class,
    provider_name: str = "OpenAI"
) -> Tuple[str, StreamingTiming]:
    """
    Stream from OpenAI-compatible LLM (OpenAI, Groq, etc.).
    
    This handles any LLM that uses the OpenAI-style streaming API.
    
    Args:
        llm: LLM with get_client() method
        messages: List of conversation messages
        start_time: Time when the request started
        serializer_class: Message serializer class (e.g., OpenAIMessageSerializer, GroqMessageSerializer)
        provider_name: Name of provider for logging
        
    Returns:
        Tuple of (completion_text, timing_info)
    """
    client = llm.get_client()
    
    # Serialize messages using the appropriate serializer
    serialized_messages = serializer_class.serialize_messages(messages)
    
    # Build model parameters (common across OpenAI-compatible APIs)
    model_params = {}
    if hasattr(llm, 'temperature') and llm.temperature is not None:
        model_params['temperature'] = llm.temperature
    if hasattr(llm, 'top_p') and llm.top_p is not None:
        model_params['top_p'] = llm.top_p
    if hasattr(llm, 'seed') and llm.seed is not None:
        model_params['seed'] = llm.seed
    
    # OpenAI-specific parameters
    if hasattr(llm, 'frequency_penalty') and llm.frequency_penalty is not None:
        model_params['frequency_penalty'] = llm.frequency_penalty
    if hasattr(llm, 'max_completion_tokens') and llm.max_completion_tokens is not None:
        model_params['max_completion_tokens'] = llm.max_completion_tokens
    
    # Create streaming request
    # Note: For some providers (e.g., Groq), the create() call with stream=True
    # returns an async generator directly, not a coroutine that needs to be awaited.
    # We need to handle both cases.
    stream_response = client.chat.completions.create(
        model=llm.model,
        messages=serialized_messages,
        stream=True,  # Enable streaming!
        **model_params
    )
    
    # Check if it's a coroutine (needs await) or already an async generator
    import inspect
    if inspect.iscoroutine(stream_response):
        stream = await stream_response
    else:
        stream = stream_response
    
    # Collect chunks and measure timing
    first_token_time = None
    chunks = []
    total_chars = 0
    
    async for chunk in stream:
        # Measure time to first token
        if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
            first_token_time = time.time() - start_time
            logger.debug(f"{provider_name} TTFT: {first_token_time:.3f}s")
        
        # Collect content
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            chunks.append(content)
            total_chars += len(content)
    
    total_time = time.time() - start_time
    completion = ''.join(chunks)
    
    # Calculate generation time (excluding TTFT)
    generation_time = total_time - first_token_time if first_token_time else total_time
    
    # Rough token estimate (4 chars per token is typical for English)
    estimated_tokens = total_chars // 4
    tokens_per_sec = estimated_tokens / generation_time if generation_time > 0 else None
    
    timing = StreamingTiming(
        total_seconds=total_time,
        time_to_first_token=first_token_time,
        generation_seconds=generation_time,
        tokens_per_second=tokens_per_sec,
        total_tokens=estimated_tokens,  # Estimated, not exact
        total_characters=total_chars
    )
    
    ttft_str = f"{first_token_time:.2f}s" if first_token_time else "N/A"
    speed_str = f"{tokens_per_sec:.1f} tok/s" if tokens_per_sec else "N/A"
    logger.debug(f"{provider_name} streaming complete: TTFT={ttft_str}, Gen={generation_time:.2f}s, Speed={speed_str}")
    
    return completion, timing


async def _stream_openai(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    start_time: float
) -> Tuple[str, StreamingTiming]:
    """Stream from OpenAI LLM."""
    return await _stream_openai_compatible(
        llm, messages, start_time,
        OpenAIMessageSerializer,
        "OpenAI"
    )


async def _stream_groq(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    start_time: float
) -> Tuple[str, StreamingTiming]:
    """Stream from Groq LLM."""
    from browser_use.llm.groq.serializer import GroqMessageSerializer
    
    return await _stream_openai_compatible(
        llm, messages, start_time,
        GroqMessageSerializer,
        "Groq"
    )
__all__ = ["stream_llm_call", "StreamingTiming"]
