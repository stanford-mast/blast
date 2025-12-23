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
import time
from typing import Any, ClassVar, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .engine import Engine
from .response import AgentHistoryListResponse, AgentReasoning, AgentScheduled, HumanRequest, HumanResponse, StopRequest

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
    HEARTBEAT = "heartbeat"  # Client heartbeat to keep connection alive
    HEARTBEAT_RESPONSE = "heartbeat_response"  # Server response to heartbeat


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
        MessageType.ERROR,
        MessageType.HEARTBEAT,
        MessageType.HEARTBEAT_RESPONSE,
    }

    @classmethod
    def from_agent_reasoning(cls, reasoning: AgentReasoning) -> "RealtimeMessage":
        """Create message from AgentReasoning event."""
        return cls(type=MessageType.AGENT_REASONING, data=reasoning.model_dump())

    @classmethod
    def from_agent_scheduled(cls, scheduled: AgentScheduled) -> "RealtimeMessage":
        """Create message from AgentScheduled event."""
        return cls(type=MessageType.AGENT_SCHEDULED, data=scheduled.model_dump())

    @classmethod
    def from_task_result(cls, result: AgentHistoryListResponse) -> "RealtimeMessage":
        """Create message from task result."""
        try:
            # Extract the data from the result
            data = result.model_dump() if result else {"history": [], "task_id": "unknown"}

            # Add the final result as a separate field
            try:
                # Try to get the final result using the final_result method if it exists
                if hasattr(result, "final_result") and callable(getattr(result, "final_result")):
                    final_result = result.final_result()
                    data["final_result"] = final_result
                else:
                    # If the method doesn't exist, try to get the last history item
                    if result and result.history and len(result.history) > 0:
                        last_item = result.history[-1]
                        if hasattr(last_item, "content") and last_item.content:
                            data["final_result"] = last_item.content
                        else:
                            data["final_result"] = "Task completed"
                    else:
                        data["final_result"] = "Task completed"
            except Exception as e:
                logger.error(f"Error extracting final result: {e}", exc_info=True)
                data["final_result"] = "Task completed with an error"

            return cls(type=MessageType.TASK_RESULT, data=data)
        except Exception as e:
            logger.error(f"Error creating task result message: {e}", exc_info=True)
            # Return a fallback error message
            return cls(type=MessageType.ERROR, data={"error": f"Failed to process task result: {str(e)}"})

    @classmethod
    def from_human_request(cls, request: HumanRequest) -> "RealtimeMessage":
        """Create message from human request."""
        return cls(type=MessageType.HUMAN_REQUEST, data=request.model_dump())

    @classmethod
    def error(cls, error: str) -> "RealtimeMessage":
        """Create error message."""
        return cls(type=MessageType.ERROR, data={"error": error})

    @classmethod
    def heartbeat_response(cls, session_id: str) -> "RealtimeMessage":
        """Create heartbeat response message."""
        return cls(
            type=MessageType.HEARTBEAT_RESPONSE,
            data={"timestamp": int(asyncio.get_event_loop().time() * 1000), "session_id": session_id},
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
        self.session_id: Optional[str] = None
        self.last_heartbeat: Optional[float] = None

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
                        await self.websocket.send_json(RealtimeMessage.from_agent_scheduled(event).model_dump())
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
                        await self.websocket.send_json(RealtimeMessage.from_agent_reasoning(event).model_dump())
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
                        logger.error(f"Error sending task result: {e}", exc_info=True)
                        await self.websocket.send_json(
                            RealtimeMessage.error(f"Error processing task result: {str(e)}").model_dump()
                        )
                        # Still mark task as completed on error
                        self.current_task_id = None
                        break

                elif isinstance(event, HumanRequest):
                    await self.websocket.send_json(RealtimeMessage.from_human_request(event).model_dump())

        except Exception as e:
            logger.error(f"Error forwarding engine events: {e}", exc_info=True)
            await self.websocket.send_json(RealtimeMessage.error(str(e)).model_dump())

    async def cleanup(self):
        """Clean up connection resources."""
        if self.current_task_id and self.queues:
            # Don't send stop request to current task
            # try:
            #     logger.info(f"Sending stop request during cleanup for task {self.current_task_id}")
            #     await self.queues["from_client"].put(StopRequest(type="stop"))
            # except Exception as e:
            #     logger.error(f"Error sending stop request during cleanup: {e}")

            # Don't set queues to None - this allows the connection to be reused
            # when the client reconnects
            logger.info(f"Preserving connection state for task {self.current_task_id}")

            # # Just mark the task as completed
            # self.current_task_id = None


# Dictionary to track connections by session ID
_session_connections: Dict[str, str] = {}  # Maps session_id to connection_id

# Heartbeat timeout in seconds
HEARTBEAT_TIMEOUT = 120  # 2 minutes


async def cleanup_stale_connections(connections: Dict[str, RealtimeConnection]):
    """Background task to clean up stale connections.

    This task runs periodically and removes connections that haven't
    received a heartbeat in HEARTBEAT_TIMEOUT seconds.

    Args:
        connections: Dictionary of active connections
    """
    while True:
        try:
            # Wait for a while before checking
            await asyncio.sleep(60)  # Check every minute

            current_time = asyncio.get_event_loop().time()
            stale_connections: List[str] = []

            # Find stale connections
            for conn_id, conn in connections.items():
                if conn.last_heartbeat:
                    time_since_heartbeat = current_time - conn.last_heartbeat
                    if time_since_heartbeat > HEARTBEAT_TIMEOUT:
                        logger.info(f"Connection {conn_id} is stale (no heartbeat for {time_since_heartbeat:.1f}s)")
                        stale_connections.append(conn_id)
                else:
                    # No heartbeat recorded, check if it's been a while since creation
                    stale_connections.append(conn_id)

            # Clean up stale connections
            for conn_id in stale_connections:
                if conn_id in connections:
                    conn = connections[conn_id]

                    # Clean up session tracking
                    if conn.session_id and conn.session_id in _session_connections:
                        del _session_connections[conn.session_id]

                    # Clean up connection
                    await conn.cleanup()
                    del connections[conn_id]

                    logger.info(f"Cleaned up stale connection {conn_id}")

        except Exception as e:
            logger.error(f"Error in cleanup_stale_connections: {e}")


async def find_existing_session(
    session_id: str, connections: Dict[str, RealtimeConnection]
) -> Optional[RealtimeConnection]:
    """Find an existing connection by session ID.

    Args:
        session_id: The session ID to look for
        connections: Dictionary of active connections

    Returns:
        The existing connection if found, None otherwise
    """
    if not session_id or session_id not in _session_connections:
        return None

    connection_id = _session_connections[session_id]
    if connection_id not in connections:
        # Clean up stale session tracking
        del _session_connections[session_id]
        return None

    return connections[connection_id]


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

    # Set last heartbeat time to now
    connection.last_heartbeat = asyncio.get_event_loop().time()

    # Flag to track if this is the first message
    is_first_message = True

    try:
        while True:
            # Receive message from client
            raw_message = await websocket.receive_text()

            message = RealtimeMessage.model_validate_json(raw_message)

            # Check for session ID in the first message
            if is_first_message:
                is_first_message = False

                # Try to extract session ID from any message type
                session_id = None
                if hasattr(message.data, "get"):
                    session_id = message.data.get("session_id")

                if session_id:
                    logger.info(f"First message contains session ID: {session_id}")

                    # Check if this is a reconnection
                    existing_connection = await find_existing_session(session_id, connections)
                    if existing_connection:
                        logger.info(f"Reconnecting session {session_id} to connection {connection_id}")

                        # Update the connection with the existing session's state
                        connection.session_id = session_id
                        connection.current_task_id = existing_connection.current_task_id
                        connection.queues = existing_connection.queues

                        # Update session tracking
                        _session_connections[session_id] = connection_id

                        # Clean up the old connection
                        old_connection_id = existing_connection.connection_id
                        if old_connection_id in connections and old_connection_id != connection_id:
                            del connections[old_connection_id]
                            logger.info(f"Cleaned up old connection {old_connection_id}")

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
                            initial_url=task_request.initial_url,  # Pass the initial URL to the engine
                        )

                        # Update connection state
                        connection.queues = {"to_client": to_client_queue, "from_client": from_client_queue}

                        # Start forwarding in background
                        asyncio.create_task(connection.forward_engine_events())
                    except Exception as e:
                        logger.error(f"Error executing task: {str(e)}", exc_info=True)
                        raise

                elif message.type == MessageType.STOP:
                    # Forward stop request to engine
                    if connection.queues:
                        try:
                            # Store task ID before sending stop request
                            task_id = connection.current_task_id
                            logger.info(f"Processing stop request for task {task_id}")

                            # Send stop request to engine
                            await connection.queues["from_client"].put(StopRequest(type="stop"))

                            # Wait a short time for the stop request to be processed
                            await asyncio.sleep(0.2)

                            try:
                                # Send confirmation back to client
                                await websocket.send_json(RealtimeMessage.error("Task stopped by user").model_dump())

                                # Log stop request but don't clear connection state
                                # This allows the connection to be reused if the executor is preserved
                                logger.info(f"Stop request processed for task {task_id}, connection state preserved")

                                # Mark the task as completed but preserve the queues
                                connection.current_task_id = None
                            except RuntimeError as e:
                                # Handle case where websocket is closed before we can send confirmation
                                if "websocket.close" in str(e):
                                    logger.warning(
                                        f"WebSocket closed before stop confirmation could be sent for task {task_id}"
                                    )
                                    # Still mark the task as completed
                                    connection.current_task_id = None
                                else:
                                    raise

                        except Exception as e:
                            logger.error(f"Error processing stop request: {e}", exc_info=True)
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
                            HumanResponse(task_id=connection.current_task_id, response=response_data["response"])
                        )

                elif message.type == MessageType.HEARTBEAT:
                    # Handle heartbeat message
                    try:
                        # Extract session ID from heartbeat message
                        heartbeat_data = message.data
                        session_id = heartbeat_data.get("session_id")

                        if session_id:
                            # Update connection's session ID if not already set
                            if not connection.session_id:
                                connection.session_id = session_id
                                logger.info(f"Connection {connection_id} associated with session {session_id}")

                                # Update session tracking
                                _session_connections[session_id] = connection_id

                            # Update last heartbeat time
                            connection.last_heartbeat = asyncio.get_event_loop().time()

                            # Send heartbeat response
                            await websocket.send_json(RealtimeMessage.heartbeat_response(session_id).model_dump())

                            logger.debug(f"Heartbeat received and acknowledged for session {session_id}")
                        else:
                            logger.warning("Received heartbeat without session_id")
                    except Exception as e:
                        logger.error(f"Error handling heartbeat: {e}")

            except ValueError as e:
                # Handle validation errors
                logger.error(f"Invalid message: {e}")
                await websocket.send_json(RealtimeMessage.error(str(e)).model_dump())
            except Exception as e:
                # Handle other errors
                logger.error(f"Error handling message: {e}", exc_info=True)
                await websocket.send_json(RealtimeMessage.error(str(e)).model_dump())

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")

        # If this connection has a session ID, preserve it for potential reconnection
        if connection.session_id:
            logger.info(f"Preserving connection state for session {connection.session_id}")
            # Don't delete the connection from connections dict yet
            # It will be cleaned up by a background task if no reconnection happens

            # Just clean up the task
            await connection.cleanup()

            # Start a background task to clean up the connection if no reconnection happens
            # within a reasonable time (e.g., 2 minutes)
            async def delayed_cleanup():
                await asyncio.sleep(120)  # 2 minutes
                if connection_id in connections:
                    # Check if there have been any heartbeats recently
                    if connection.last_heartbeat:
                        time_since_heartbeat = asyncio.get_event_loop().time() - connection.last_heartbeat
                        if time_since_heartbeat > 120:  # 2 minutes
                            logger.info(
                                f"No reconnection for session {connection.session_id} after 2 minutes, cleaning up"
                            )
                            if connection_id in connections:
                                del connections[connection_id]
                            # Clean up session tracking
                            if connection.session_id in _session_connections:
                                del _session_connections[connection.session_id]

            # Start the delayed cleanup task
            asyncio.create_task(delayed_cleanup())
        else:
            # No session ID, clean up immediately
            logger.info(f"No session ID, cleaning up connection {connection_id}")
            await connection.cleanup()
            if connection_id in connections:
                del connections[connection_id]

    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
        try:
            await websocket.send_json(RealtimeMessage.error(str(e)).model_dump())
        except:
            pass
        # Ensure cleanup
        await connection.cleanup()

        # Clean up connection from tracking
        if connection_id in connections:
            del connections[connection_id]

        # Clean up session tracking if needed
        if connection.session_id and connection.session_id in _session_connections:
            del _session_connections[connection.session_id]

    finally:
        # We don't automatically delete the connection here anymore
        # It's either already deleted or preserved for reconnection
        pass
