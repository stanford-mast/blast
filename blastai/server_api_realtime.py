"""Realtime WebSocket API models and handlers.

This module implements a WebSocket-based API for realtime task execution.
It enables:
- Interactive task execution
- Task stopping
- Human-in-the-loop communication
- Task chaining with prerequisites

The API uses JSON messages with a standard format:
{
    "type": "message_type",
    "data": { ... }
}
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, ClassVar
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .response import (
    AgentReasoning,
    AgentHistoryListResponse,
    AgentScheduled,
    HumanRequest,
    HumanResponse,
    StopRequest
)
from .engine import Engine

# Get logger for this module
logger = logging.getLogger(__name__)

class MessageType(str):
    """Types of messages that can be sent over the realtime API."""
    TASK = "task"  # New task request
    AGENT_SCHEDULED = "agent_scheduled"  # Task acknowledgment with task ID
    STOP = "stop"  # Stop current task
    HUMAN_RESPONSE = "human_response"  # Response to human-in-the-loop request
    AGENT_REASONING = "agent_reasoning"  # Agent reasoning/screenshot update
    TASK_RESULT = "task_result"  # Task completion result
    HUMAN_REQUEST = "human_request"  # Request for human input
    ERROR = "error"  # Error message

class TaskRequest(BaseModel):
    """Request to run a new task.
    
    Attributes:
        description: Task description/instructions
        initial_url: Optional starting URL for browser tasks
        prerequisite_task_id: Optional ID of task that must complete first
    """
    description: str = Field(..., description="Task description/instructions")
    initial_url: Optional[str] = Field(None, description="Optional starting URL for browser tasks")
    prerequisite_task_id: Optional[str] = Field(None, description="Optional ID of task that must complete first")

class RealtimeMessage(BaseModel):
    """Base message format for realtime API communication.
    
    All messages follow this format with a type and data payload.
    The data schema varies based on the message type.
    
    Attributes:
        type: Message type from MessageType
        data: Message payload, schema depends on type
    """
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message payload")

    # Valid message types
    VALID_TYPES: ClassVar[set] = {
        MessageType.TASK,
        MessageType.AGENT_SCHEDULED,
        MessageType.STOP,
        MessageType.HUMAN_RESPONSE,
        MessageType.AGENT_REASONING,
        MessageType.TASK_RESULT,
        MessageType.HUMAN_REQUEST,
        MessageType.ERROR
    }

    @classmethod
    def from_agent_reasoning(cls, reasoning: AgentReasoning) -> "RealtimeMessage":
        """Create message from AgentReasoning event."""
        return cls(
            type=MessageType.AGENT_REASONING,
            data=reasoning.model_dump()
        )
        
    @classmethod
    def from_agent_scheduled(cls, scheduled: AgentScheduled) -> "RealtimeMessage":
        """Create message from AgentScheduled event."""
        return cls(
            type=MessageType.AGENT_SCHEDULED,
            data=scheduled.model_dump()
        )

    @classmethod
    def from_task_result(cls, result: AgentHistoryListResponse) -> "RealtimeMessage":
        """Create message from task result."""
        try:
            # Extract the data from the result
            data = result.model_dump() if result else {"history": [], "task_id": "unknown"}
            
            # Add the final result as a separate field
            try:
                # Try to get the final result using the final_result method if it exists
                if hasattr(result, 'final_result') and callable(getattr(result, 'final_result')):
                    final_result = result.final_result()
                    data["final_result"] = final_result
                else:
                    # If the method doesn't exist, try to get the last history item
                    if result and result.history and len(result.history) > 0:
                        last_item = result.history[-1]
                        if hasattr(last_item, 'content') and last_item.content:
                            data["final_result"] = last_item.content
                        else:
                            data["final_result"] = "Task completed"
                    else:
                        data["final_result"] = "Task completed"
            except Exception as e:
                logger.error(f"Error extracting final result: {e}")
                data["final_result"] = "Task completed with an error"
            
            return cls(
                type=MessageType.TASK_RESULT,
                data=data
            )
        except Exception as e:
            logger.error(f"Error creating task result message: {e}")
            # Return a fallback error message
            return cls(
                type=MessageType.ERROR,
                data={"error": f"Failed to process task result: {str(e)}"}
            )

    @classmethod
    def from_human_request(cls, request: HumanRequest) -> "RealtimeMessage":
        """Create message from human request."""
        return cls(
            type=MessageType.HUMAN_REQUEST,
            data=request.model_dump()
        )

    @classmethod
    def error(cls, error: str) -> "RealtimeMessage":
        """Create error message."""
        return cls(
            type=MessageType.ERROR,
            data={"error": error}
        )

    def validate_type(self) -> None:
        """Validate message type."""
        if self.type not in self.VALID_TYPES:
            raise ValueError(f"Invalid message type: {self.type}")

class RealtimeConnection:
    """Manages state and handlers for a realtime WebSocket connection.
    
    This class encapsulates:
    - WebSocket connection
    - Current task state
    - Communication queues
    - Event forwarding
    
    Attributes:
        websocket: The WebSocket connection
        connection_id: Unique connection identifier
        current_task_id: ID of currently running task
        queues: Communication queues for current task
    """
    
    def __init__(self, websocket: WebSocket, connection_id: str):
        """Initialize connection state.
        
        Args:
            websocket: The WebSocket connection
            connection_id: Unique connection identifier
        """
        self.websocket = websocket
        self.connection_id = connection_id
        self.current_task_id: Optional[str] = None
        self.queues: Optional[Dict[str, asyncio.Queue]] = None
        
    async def forward_engine_events(self):
        """Forward events from engine to WebSocket.
        
        Handles:
        - Agent reasoning/screenshot updates
        - Task completion results
        - Human-in-the-loop requests
        """
        if not self.queues:
            return
            
        try:
            websocket_active = True
            while websocket_active:
                event = await self.queues["to_client"].get()
                
                if isinstance(event, AgentScheduled):
                    # Store task ID from scheduled event
                    self.current_task_id = event.task_id
                    
                    try:
                        await self.websocket.send_json(
                            RealtimeMessage.from_agent_scheduled(event).model_dump()
                        )
                    except RuntimeError as e:
                        if "websocket.close" in str(e):
                            websocket_active = False
                            break
                        raise
                
                elif isinstance(event, AgentReasoning):
                    # Store current task ID from first event if not already set
                    if not self.current_task_id:
                        self.current_task_id = event.task_id
                        
                    try:
                        await self.websocket.send_json(
                            RealtimeMessage.from_agent_reasoning(event).model_dump()
                        )
                    except RuntimeError as e:
                        if "websocket.close" in str(e):
                            websocket_active = False
                            break
                        raise
                    
                elif isinstance(event, AgentHistoryListResponse):
                    try:
                        result_message = RealtimeMessage.from_task_result(event)
                        await self.websocket.send_json(result_message.model_dump())
                        # Task completed, but don't clear queues - just mark task as completed
                        logger.info(f"Task {self.current_task_id} completed, preserving connection state")
                        self.current_task_id = None
                        break
                    except Exception as e:
                        logger.error(f"Error sending task result: {e}")
                        await self.websocket.send_json(
                            RealtimeMessage.error(f"Error processing task result: {str(e)}").model_dump()
                        )
                        # Still mark task as completed on error
                        self.current_task_id = None
                        break
                    
                elif isinstance(event, HumanRequest):
                    await self.websocket.send_json(
                        RealtimeMessage.from_human_request(event).model_dump()
                    )
                    
        except Exception as e:
            logger.error(f"Error forwarding engine events: {e}")
            await self.websocket.send_json(RealtimeMessage.error(str(e)).model_dump())

    async def cleanup(self):
        """Clean up connection resources."""
        if self.current_task_id and self.queues:
            # Send stop request to current task
            try:
                logger.info(f"Sending stop request during cleanup for task {self.current_task_id}")
                await self.queues["from_client"].put(StopRequest(type="stop"))
            except Exception as e:
                logger.error(f"Error sending stop request during cleanup: {e}")
            
            # Don't set queues to None - this allows the connection to be reused
            # when the client reconnects
            logger.info(f"Preserving connection state for task {self.current_task_id}")
            
            # Just mark the task as completed
            self.current_task_id = None

async def handle_realtime_connection(websocket: WebSocket, engine: Engine, connections: Dict[str, RealtimeConnection]):
    """Handle a realtime WebSocket connection.
    
    This handler:
    1. Accepts the WebSocket connection
    2. Creates connection state
    3. Processes incoming messages
    4. Forwards engine events
    5. Handles cleanup on disconnect
    
    Args:
        websocket: The WebSocket connection
        engine: The engine instance
        connections: Dictionary of active connections
    """
    # Accept connection
    await websocket.accept()
    
    # Create connection state
    connection_id = f"conn_{len(connections)}"
    connection = RealtimeConnection(websocket, connection_id)
    connections[connection_id] = connection
    
    # No ping handler
    
    try:
        while True:
            # Receive message from client
            raw_message = await websocket.receive_text()
            
            message = RealtimeMessage.model_validate_json(raw_message)
            
            try:
                # Validate message type
                message.validate_type()
                
                if message.type == MessageType.TASK:
                    # Handle new task request
                    task_request = TaskRequest.model_validate(message.data)
                    
                    try:
                        # Run task in interactive mode
                        to_client_queue, from_client_queue = await engine.run(
                            task_descriptions=task_request.description,
                            mode="interactive",  # This automatically disables caching
                            previous_response_id=task_request.prerequisite_task_id,
                            initial_url=task_request.initial_url  # Pass the initial URL to the engine
                        )
                        
                        # Update connection state
                        connection.queues = {
                            "to_client": to_client_queue,
                            "from_client": from_client_queue
                        }
                        
                        # Start forwarding in background
                        asyncio.create_task(connection.forward_engine_events())
                    except Exception as e:
                        logger.error(f"Error executing task: {str(e)}")
                        raise
                    
                elif message.type == MessageType.STOP:
                    # Forward stop request to engine
                    if connection.queues:
                        try:
                            # Store task ID before sending stop request
                            task_id = connection.current_task_id
                            logger.info(f"Processing stop request for task {task_id}")
                            
                            # Send stop request to engine
                            await connection.queues["from_client"].put(
                                StopRequest(type="stop")
                            )
                            
                            # Wait a short time for the stop request to be processed
                            await asyncio.sleep(0.2)
                            
                            try:
                                # Send confirmation back to client
                                await websocket.send_json(
                                    RealtimeMessage.error("Task stopped by user").model_dump()
                                )
                                
                                # Log stop request but don't clear connection state
                                # This allows the connection to be reused if the executor is preserved
                                logger.info(f"Stop request processed for task {task_id}, connection state preserved")
                                
                                # Mark the task as completed but preserve the queues
                                connection.current_task_id = None
                            except RuntimeError as e:
                                # Handle case where websocket is closed before we can send confirmation
                                if "websocket.close" in str(e):
                                    logger.warning(f"WebSocket closed before stop confirmation could be sent for task {task_id}")
                                    # Still mark the task as completed
                                    connection.current_task_id = None
                                else:
                                    raise
                            
                        except Exception as e:
                            logger.error(f"Error processing stop request: {e}")
                            try:
                                await websocket.send_json(
                                    RealtimeMessage.error(f"Error stopping task: {str(e)}").model_dump()
                                )
                            except RuntimeError as ws_err:
                                # Handle case where websocket is closed before we can send error
                                if "websocket.close" in str(ws_err):
                                    logger.warning(f"WebSocket closed before error could be sent: {e}")
                                else:
                                    raise
                        
                elif message.type == MessageType.HUMAN_RESPONSE:
                    # Forward human response to engine
                    if connection.queues and connection.current_task_id:
                        response_data = message.data
                        await connection.queues["from_client"].put(
                            HumanResponse(
                                task_id=connection.current_task_id,
                                response=response_data["response"]
                            )
                        )
                        
            except ValueError as e:
                # Handle validation errors
                logger.error(f"Invalid message: {e}")
                await websocket.send_json(RealtimeMessage.error(str(e)).model_dump())
            except Exception as e:
                # Handle other errors
                logger.error(f"Error handling message: {e}")
                await websocket.send_json(RealtimeMessage.error(str(e)).model_dump())
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
        # Clean up connection
        await connection.cleanup()
        if connection_id in connections:
            del connections[connection_id]
            
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        try:
            await websocket.send_json(RealtimeMessage.error(str(e)).model_dump())
        except:
            pass
        # Ensure cleanup
        await connection.cleanup()
            
    finally:
        # Final cleanup
        if connection_id in connections:
            del connections[connection_id]