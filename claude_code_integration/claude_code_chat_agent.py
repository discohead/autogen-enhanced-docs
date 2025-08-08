"""Claude Code Chat Agent for AutoGen.

This module provides a custom ChatAgent implementation that integrates
Claude Code SDK with AutoGen's agent system, preserving Claude Code's
native tool execution while supporting AutoGen's team coordination.
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Sequence, Union
from dataclasses import dataclass, field

import anyio
from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    AssistantMessage as ClaudeAssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
    SystemMessage as ClaudeSystemMessage,
    UserMessage as ClaudeUserMessage,
    Message as ClaudeMessage,
)

from autogen_agentchat.base import BaseChatAgent, Response
from autogen_agentchat.messages import (
    BaseChatMessage,
    BaseAgentEvent,
    TextMessage,
    MultiModalMessage,
    HandoffMessage,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
    ModelClientStreamingChunkEvent,
    StopMessage,
)
from autogen_core import CancellationToken, FunctionCall, FunctionExecutionResult
from pydantic import BaseModel, ConfigDict

# Event logger for AutoGen
EVENT_LOGGER_NAME = "autogen_agentchat.events"
event_logger = logging.getLogger(EVENT_LOGGER_NAME)


@dataclass
class ClaudeCodeAgentState:
    """State for ClaudeCodeChatAgent."""
    conversation_history: List[BaseChatMessage] = field(default_factory=list)
    tool_execution_history: List[Dict[str, Any]] = field(default_factory=list)
    session_id: Optional[str] = None


class ClaudeCodeChatAgent(BaseChatAgent):
    """A chat agent that uses Claude Code SDK for local Claude interface.
    
    This agent integrates Claude Code's powerful tool system (Read, Write, Bash, etc.)
    with AutoGen's multi-agent coordination. Unlike using a ChatCompletionClient,
    this agent:
    
    1. Preserves Claude Code's native tool execution
    2. Emits proper events for tool usage visibility
    3. Supports handoffs for team coordination
    4. Maintains conversation state across interactions
    
    Args:
        name: The agent's name (must be unique within a team)
        description: Description of the agent's capabilities
        system_prompt: System instructions for Claude
        allowed_tools: List of Claude Code tools to enable (e.g., ["Read", "Write", "Bash"])
        permission_mode: How to handle tool permissions ("default", "acceptEdits", "bypassPermissions")
        max_turns: Maximum conversation turns per task
        cwd: Working directory for Claude Code operations
        model: The model to use (default: claude-3-5-sonnet-20241022)
        handoffs: List of agents this agent can hand off to
        emit_tool_events: Whether to emit events for tool usage (default: True)
        **kwargs: Additional options passed to ClaudeCodeOptions
    
    Example:
        ```python
        agent = ClaudeCodeChatAgent(
            name="coder",
            description="Writes and edits code files",
            allowed_tools=["Read", "Write", "Edit"],
            permission_mode="acceptEdits",
            system_prompt="You are an expert Python developer."
        )
        
        # Use in a team
        team = RoundRobinGroupChat([agent, reviewer_agent])
        result = await team.run(task="Create a hello world script")
        ```
    """
    
    def __init__(
        self,
        name: str,
        description: str = "A Claude Code powered agent",
        system_prompt: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        permission_mode: Optional[str] = None,
        max_turns: Optional[int] = None,
        cwd: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        handoffs: Optional[List[Union[str, Dict[str, str]]]] = None,
        emit_tool_events: bool = True,
        **kwargs: Any
    ):
        super().__init__(name=name, description=description)
        
        self._system_prompt = system_prompt
        self._allowed_tools = allowed_tools or []
        self._permission_mode = permission_mode
        self._max_turns = max_turns
        self._cwd = cwd
        self._model = model
        self._handoffs = handoffs or []
        self._emit_tool_events = emit_tool_events
        self._additional_options = kwargs
        
        # State management
        self._state = ClaudeCodeAgentState()
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """Message types this agent can produce."""
        return [TextMessage, HandoffMessage, StopMessage]
    
    def _format_handoff_prompt(self) -> str:
        """Format handoff instructions for the system prompt."""
        if not self._handoffs:
            return ""
        
        handoff_lines = ["", "You can hand off conversations to other agents:"]
        for handoff in self._handoffs:
            if isinstance(handoff, str):
                handoff_lines.append(f"- Hand off to '{handoff}' when appropriate")
            else:
                target = handoff.get("target", "unknown")
                description = handoff.get("description", "")
                handoff_lines.append(f"- Hand off to '{target}': {description}")
        
        handoff_lines.append("\nTo hand off, say 'HANDOFF: <agent_name>' followed by context for the next agent.")
        return "\n".join(handoff_lines)
    
    def _build_claude_prompt(self, messages: Sequence[BaseChatMessage]) -> str:
        """Convert AutoGen messages to Claude prompt format."""
        prompt_parts = []
        
        for msg in messages:
            if isinstance(msg, TextMessage):
                # Determine if this is from a user or another agent
                if msg.source == "user" or msg.source not in ["assistant", self.name]:
                    prompt_parts.append(f"Human: {msg.content}")
                else:
                    prompt_parts.append(f"Assistant: {msg.content}")
            elif isinstance(msg, MultiModalMessage):
                # For now, just extract text content
                text_content = " ".join(str(c) for c in msg.content if isinstance(c, str))
                prompt_parts.append(f"Human: {text_content}")
            elif isinstance(msg, HandoffMessage):
                prompt_parts.append(f"Human: [Handoff from {msg.source} to {msg.target}] {msg.content}")
        
        prompt = "\n\n".join(prompt_parts)
        
        # Ensure the prompt ends with Human: for Claude
        if prompt_parts and not prompt_parts[-1].startswith("Human:"):
            prompt += "\n\nHuman: Please continue."
        
        return prompt
    
    def _parse_for_handoff(self, text: str) -> Optional[Dict[str, str]]:
        """Check if the text contains a handoff request."""
        if "HANDOFF:" in text:
            parts = text.split("HANDOFF:", 1)
            if len(parts) > 1:
                # Extract target agent name
                remaining = parts[1].strip()
                target = remaining.split()[0].strip() if remaining else None
                
                if target:
                    # Extract context after agent name
                    context = remaining[len(target):].strip()
                    return {"target": target, "context": context}
        return None
    
    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken
    ) -> Response:
        """Process messages and return a response."""
        # Convert messages to Claude format
        prompt = self._build_claude_prompt(messages)
        
        # Build system prompt with handoff instructions
        system_prompt = self._system_prompt or ""
        if self._handoffs:
            system_prompt += self._format_handoff_prompt()
        
        # Create Claude options
        options = ClaudeCodeOptions(
            model=self._model,
            system_prompt=system_prompt,
            allowed_tools=self._allowed_tools,
            permission_mode=self._permission_mode,  # type: ignore
            max_turns=self._max_turns,
            cwd=self._cwd,
            **self._additional_options
        )
        
        # Collect inner messages and tool events
        inner_messages: List[Union[BaseAgentEvent, BaseChatMessage]] = []
        response_text = ""
        tool_calls: List[Dict[str, Any]] = []
        
        # Query Claude Code
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, ClaudeAssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
                    elif isinstance(block, ToolUseBlock):
                        # Track tool usage
                        tool_call = {
                            "id": block.id,
                            "name": block.name,
                            "arguments": block.input
                        }
                        tool_calls.append(tool_call)
                        
                        # Emit tool request event if enabled
                        if self._emit_tool_events:
                            event = ToolCallRequestEvent(
                                source=self.name,
                                content=[FunctionCall(
                                    id=block.id,
                                    name=block.name,
                                    arguments=json.dumps(block.input) if block.input else "{}"
                                )]
                            )
                            inner_messages.append(event)
                    elif isinstance(block, ToolResultBlock):
                        # Note: This might appear in continued conversations
                        # Tool was already executed by Claude Code
                        if self._emit_tool_events:
                            # Find the corresponding tool call
                            tool_call = next((tc for tc in tool_calls if tc["id"] == block.tool_use_id), None)
                            if tool_call:
                                event = ToolCallExecutionEvent(
                                    source=self.name,
                                    content=[FunctionExecutionResult(
                                        call_id=block.tool_use_id,
                                        content=str(block.content) if block.content else "Executed",
                                        error=block.is_error
                                    )]
                                )
                                inner_messages.append(event)
            elif isinstance(msg, ResultMessage):
                # Store session info if available
                if msg.session_id:
                    self._state.session_id = msg.session_id
        
        # Check for handoff
        handoff_info = self._parse_for_handoff(response_text)
        if handoff_info:
            # Create handoff message
            chat_message = HandoffMessage(
                source=self.name,
                target=handoff_info["target"],
                content=handoff_info.get("context", response_text)
            )
        else:
            # Regular text response
            chat_message = TextMessage(
                source=self.name,
                content=response_text.strip()
            )
        
        # Update state
        self._state.conversation_history.extend(messages)
        self._state.conversation_history.append(chat_message)
        if tool_calls:
            self._state.tool_execution_history.extend(tool_calls)
        
        return Response(
            chat_message=chat_message,
            inner_messages=inner_messages if inner_messages else None
        )
    
    def on_messages_stream(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken
    ) -> AsyncGenerator[Union[BaseAgentEvent, BaseChatMessage, Response], None]:
        """Stream messages and events as they are generated."""
        async def _stream_messages():
            # Convert messages to Claude format
            prompt = self._build_claude_prompt(messages)
            
            # Build system prompt
            system_prompt = self._system_prompt or ""
            if self._handoffs:
                system_prompt += self._format_handoff_prompt()
            
            # Create Claude options
            options = ClaudeCodeOptions(
                model=self._model,
                system_prompt=system_prompt,
                allowed_tools=self._allowed_tools,
                permission_mode=self._permission_mode,  # type: ignore
                max_turns=self._max_turns,
                cwd=self._cwd,
                **self._additional_options
            )
            
            # Track state during streaming
            response_text = ""
            tool_calls: List[Dict[str, Any]] = []
            inner_messages: List[Union[BaseAgentEvent, BaseChatMessage]] = []
            
            # Stream from Claude
            async for msg in query(prompt=prompt, options=options):
                if isinstance(msg, ClaudeAssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            # Stream text chunks
                            if block.text:
                                chunk_event = ModelClientStreamingChunkEvent(
                                    source=self.name,
                                    content=block.text
                                )
                                yield chunk_event
                                response_text += block.text
                                
                        elif isinstance(block, ToolUseBlock):
                            # Emit tool request event
                            tool_call = {
                                "id": block.id,
                                "name": block.name,
                                "arguments": block.input
                            }
                            tool_calls.append(tool_call)
                            
                            if self._emit_tool_events:
                                event = ToolCallRequestEvent(
                                    source=self.name,
                                    content=[FunctionCall(
                                        id=block.id,
                                        name=block.name,
                                        arguments=json.dumps(block.input) if block.input else "{}"
                                    )]
                                )
                                yield event
                                inner_messages.append(event)
                                
                        elif isinstance(block, ToolResultBlock):
                            # Tool execution result
                            if self._emit_tool_events:
                                tool_call = next((tc for tc in tool_calls if tc["id"] == block.tool_use_id), None)
                                if tool_call:
                                    event = ToolCallExecutionEvent(
                                        source=self.name,
                                        content=[FunctionExecutionResult(
                                            call_id=block.tool_use_id,
                                            content=str(block.content) if block.content else "Executed",
                                            error=block.is_error
                                        )]
                                    )
                                    yield event
                                    inner_messages.append(event)
                                    
                elif isinstance(msg, ResultMessage):
                    # Store session info
                    if msg.session_id:
                        self._state.session_id = msg.session_id
            
            # Create final message
            handoff_info = self._parse_for_handoff(response_text)
            if handoff_info:
                chat_message = HandoffMessage(
                    source=self.name,
                    target=handoff_info["target"],
                    content=handoff_info.get("context", response_text)
                )
            else:
                chat_message = TextMessage(
                    source=self.name,
                    content=response_text.strip()
                )
            
            # Update state
            self._state.conversation_history.extend(messages)
            self._state.conversation_history.append(chat_message)
            if tool_calls:
                self._state.tool_execution_history.extend(tool_calls)
            
            # Yield final response
            yield Response(
                chat_message=chat_message,
                inner_messages=inner_messages if inner_messages else None
            )
        
        return _stream_messages()
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent to initial state."""
        self._state = ClaudeCodeAgentState()
    
    async def on_pause(self, cancellation_token: CancellationToken) -> None:
        """Pause the agent (no-op for this implementation)."""
        pass
    
    async def on_resume(self, cancellation_token: CancellationToken) -> None:
        """Resume the agent (no-op for this implementation)."""
        pass
    
    async def save_state(self) -> Mapping[str, Any]:
        """Save the agent's state for later restoration."""
        return {
            "conversation_history": [msg.model_dump() for msg in self._state.conversation_history],
            "tool_execution_history": self._state.tool_execution_history,
            "session_id": self._state.session_id
        }
    
    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore the agent from saved state."""
        # Note: This is simplified - in production you'd need to deserialize messages properly
        self._state.conversation_history = state.get("conversation_history", [])
        self._state.tool_execution_history = state.get("tool_execution_history", [])
        self._state.session_id = state.get("session_id")
    
    async def close(self) -> None:
        """Release any resources (no-op for Claude Code SDK)."""
        pass