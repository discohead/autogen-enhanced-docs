"""Claude Code SDK model client for AutoGen.

This module provides a custom ChatCompletionClient implementation that uses
the Claude Code SDK instead of making direct API calls, allowing you to
interface with your local Claude Code installation.
"""

import json
from typing import Any, AsyncGenerator, List, Optional, Sequence, Union
from collections.abc import AsyncIterable

import anyio
from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ResultMessage,
    Message as ClaudeMessage,
)

from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    ModelInfo,
    ModelCapabilities,
    FinishReasons,
    RequestUsage,
)
from autogen_core.messages import (
    LLMMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage as AutoGenAssistantMessage,
    FunctionCall,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
)


class ClaudeCodeChatCompletionClient(ChatCompletionClient):
    """A ChatCompletionClient that uses Claude Code SDK for local Claude interface.
    
    This client allows you to use your local Claude Code installation with AutoGen
    agents, providing a seamless integration between the two systems.
    
    Args:
        model: The model name (passed to Claude Code if specified in options)
        system_prompt: Default system prompt for all conversations
        allowed_tools: List of tools Claude Code can use (e.g., ["Read", "Write", "Bash"])
        permission_mode: Permission mode for tool usage ("default", "acceptEdits", "bypassPermissions")
        max_turns: Maximum conversation turns
        cwd: Working directory for Claude Code operations
        **kwargs: Additional options passed to ClaudeCodeOptions
    """
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        system_prompt: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        permission_mode: Optional[str] = None,
        max_turns: Optional[int] = None,
        cwd: Optional[str] = None,
        **kwargs: Any
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools or []
        self.permission_mode = permission_mode
        self.max_turns = max_turns
        self.cwd = cwd
        self.additional_options = kwargs
        
        # Track conversation history for context
        self._conversation_history: List[LLMMessage] = []
    
    @property
    def model_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            vision=True,  # Claude supports vision
            function_calling=True,  # Claude Code SDK supports tools
            json_output=False,  # Not directly supported
            streaming=True,  # Streaming is supported
            capabilities=ModelCapabilities(
                vision=True,
                function_calling=True,
                json_output=False,
            )
        )
    
    def _convert_to_claude_prompt(self, messages: Sequence[LLMMessage]) -> str:
        """Convert AutoGen messages to a Claude prompt string."""
        prompt_parts = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System messages are handled separately in ClaudeCodeOptions
                continue
            elif isinstance(msg, UserMessage):
                prompt_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AutoGenAssistantMessage):
                prompt_parts.append(f"Assistant: {msg.content}")
            elif isinstance(msg, FunctionExecutionResultMessage):
                # Convert function results to a format Claude can understand
                for result in msg.content:
                    prompt_parts.append(
                        f"Function {result.call_id} result: {result.content}"
                    )
        
        # Join all parts and ensure it ends with a Human message
        prompt = "\n\n".join(prompt_parts)
        if not prompt.endswith("Human:") and not prompt.split("\n")[-1].startswith("Human:"):
            # If the last message isn't from Human, add a continuation prompt
            prompt += "\n\nHuman: Please continue."
        
        return prompt
    
    def _extract_system_prompt(self, messages: Sequence[LLMMessage]) -> Optional[str]:
        """Extract system prompt from messages."""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                return msg.content
        return self.system_prompt
    
    def _parse_claude_response(self, claude_messages: List[ClaudeMessage]) -> CreateResult:
        """Parse Claude Code SDK response into AutoGen CreateResult."""
        content = ""
        function_calls = []
        usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        cost = 0.0
        
        for msg in claude_messages:
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        content += block.text
                    elif isinstance(block, ToolUseBlock):
                        # Convert tool use to function call
                        function_calls.append(
                            FunctionCall(
                                id=block.id,
                                name=block.name,
                                arguments=json.dumps(block.input) if block.input else "{}"
                            )
                        )
            elif isinstance(msg, ResultMessage):
                # Extract usage information
                if msg.usage:
                    usage = RequestUsage(
                        prompt_tokens=msg.usage.get("input_tokens", 0),
                        completion_tokens=msg.usage.get("output_tokens", 0),
                    )
                if msg.total_cost_usd:
                    cost = msg.total_cost_usd
        
        # Create response message
        response_message = AutoGenAssistantMessage(
            content=content.strip() if content else "",
            function_calls=function_calls if function_calls else None,
        )
        
        return CreateResult(
            content=response_message.content,
            finish_reason=FinishReasons.Stop,
            usage=usage,
            cost=cost,
            function_calls=function_calls if function_calls else None,
        )
    
    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Any] = None,
        **kwargs: Any
    ) -> CreateResult:
        """Create a completion using Claude Code SDK.
        
        Args:
            messages: The conversation messages
            tools: Optional tools (not used - tools are configured at client level)
            tool_choice: Optional tool choice (not used)
            response_format: Optional response format (not used)
            **kwargs: Additional arguments
            
        Returns:
            CreateResult with the Claude response
        """
        # Convert messages to Claude prompt
        prompt = self._convert_to_claude_prompt(messages)
        system_prompt = self._extract_system_prompt(messages)
        
        # Build options
        options = ClaudeCodeOptions(
            model=self.model,
            system_prompt=system_prompt,
            allowed_tools=self.allowed_tools,
            permission_mode=self.permission_mode,  # type: ignore
            max_turns=self.max_turns,
            cwd=self.cwd,
            **self.additional_options
        )
        
        # Collect all messages from Claude
        claude_messages: List[ClaudeMessage] = []
        async for msg in query(prompt=prompt, options=options):
            claude_messages.append(msg)
        
        # Parse and return result
        return self._parse_claude_response(claude_messages)
    
    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Any] = None,
        **kwargs: Any
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        """Create a streaming completion using Claude Code SDK.
        
        Args:
            messages: The conversation messages
            tools: Optional tools (not used - tools are configured at client level)
            tool_choice: Optional tool choice (not used)
            response_format: Optional response format (not used)
            **kwargs: Additional arguments
            
        Yields:
            String chunks of the response as they arrive, then final CreateResult
        """
        # Convert messages to Claude prompt
        prompt = self._convert_to_claude_prompt(messages)
        system_prompt = self._extract_system_prompt(messages)
        
        # Build options
        options = ClaudeCodeOptions(
            model=self.model,
            system_prompt=system_prompt,
            allowed_tools=self.allowed_tools,
            permission_mode=self.permission_mode,  # type: ignore
            max_turns=self.max_turns,
            cwd=self.cwd,
            **self.additional_options
        )
        
        # Stream messages from Claude
        collected_text = ""
        function_calls = []
        usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        cost = 0.0
        
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        # Yield text chunks as they come
                        yield block.text
                        collected_text += block.text
                    elif isinstance(block, ToolUseBlock):
                        function_calls.append(
                            FunctionCall(
                                id=block.id,
                                name=block.name,
                                arguments=json.dumps(block.input) if block.input else "{}"
                            )
                        )
            elif isinstance(msg, ResultMessage):
                # Extract final usage information
                if msg.usage:
                    usage = RequestUsage(
                        prompt_tokens=msg.usage.get("input_tokens", 0),
                        completion_tokens=msg.usage.get("output_tokens", 0),
                    )
                if msg.total_cost_usd:
                    cost = msg.total_cost_usd
        
        # Yield final CreateResult
        yield CreateResult(
            content=collected_text.strip() if collected_text else "",
            finish_reason=FinishReasons.Stop,
            usage=usage,
            cost=cost,
            function_calls=function_calls if function_calls else None,
        )
    
    async def actual_usage(self) -> RequestUsage:
        """Get actual token usage (returns last known usage)."""
        # This would need to track usage across calls if needed
        return RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    async def total_usage(self) -> RequestUsage:
        """Get total token usage (returns last known usage)."""
        # This would need to track usage across calls if needed
        return RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    async def actual_cost(self) -> float:
        """Get actual cost (returns 0.0 as cost tracking happens per request)."""
        return 0.0
    
    async def total_cost(self) -> float:
        """Get total cost (returns 0.0 as cost tracking happens per request)."""
        return 0.0
    
    async def remaining_budget(self) -> float:
        """Get remaining budget (returns infinity as there's no budget limit)."""
        return float("inf")
    
    async def close(self) -> None:
        """Close the client (no-op for Claude Code SDK)."""
        pass
    
    async def __aenter__(self) -> "ClaudeCodeChatCompletionClient":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()