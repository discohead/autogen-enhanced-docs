# Claude Code SDK Integration with AutoGen

This guide demonstrates how to integrate the Claude Code SDK as a custom model provider for AutoGen agents.

## Overview

The integration consists of:
- `claude_code_model_client.py` - A custom `ChatCompletionClient` implementation
- `example_claude_code_agent.py` - Comprehensive examples of using the integration

## Installation

### Prerequisites

1. Install AutoGen:
```bash
pip install autogen-agentchat autogen-ext
```

2. Install the Claude Code SDK (when available):
```bash
# pip install claude-code-sdk  # Replace with actual package name
```

### Setup

1. Set your Claude Code API key:
```bash
export CLAUDE_CODE_API_KEY="your-api-key-here"
```

2. Place the integration files in your project:
```
your_project/
├── claude_code_model_client.py
├── example_claude_code_agent.py
└── your_app.py
```

## Usage

### Basic Agent

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from claude_code_model_client import ClaudeCodeChatCompletionClient

async def main():
    # Create model client
    model_client = ClaudeCodeChatCompletionClient(
        model="claude-code",
        temperature=0.7,
        max_tokens=2000
    )
    
    # Create agent
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful AI assistant."
    )
    
    # Run task
    result = await agent.run(task="Explain Python decorators")
    print(result.messages[-1].content)
    
    # Cleanup
    await model_client.close()

asyncio.run(main())
```

### Agent with Tools

```python
from autogen_core.tools import FunctionTool

async def web_search(query: str) -> str:
    return f"Search results for: {query}"

tool = FunctionTool(web_search, description="Search the web")

agent = AssistantAgent(
    name="researcher",
    model_client=model_client,
    tools=[tool]
)
```

### Multi-Agent Team

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

# Create specialized agents
analyst = AssistantAgent(
    "analyst",
    model_client=model_client,
    system_message="You analyze data and provide insights."
)

writer = AssistantAgent(
    "writer",
    model_client=model_client,
    system_message="You write clear reports."
)

# Create team
team = RoundRobinGroupChat(
    participants=[analyst, writer],
    termination_condition=MaxMessageTermination(max_messages=6)
)

# Run collaborative task
result = await team.run(task="Analyze and report on AI trends")
```

### Streaming Responses

```python
agent = AssistantAgent(
    name="streaming_agent",
    model_client=model_client,
    model_client_stream=True  # Enable streaming
)

async for chunk in agent.run_stream(task="Write a story"):
    # Handle streaming chunks
    print(chunk, end="", flush=True)
```

### Structured Output

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    key_points: list[str]
    recommendation: str

result = await agent.run(
    task="Analyze this business proposal",
    json_output=Analysis  # Request structured output
)

# Parse response
analysis = Analysis.model_validate_json(result.messages[-1].content)
```

## Key Features

### 1. Message Type Conversion
The client automatically converts between AutoGen and Claude Code SDK message formats:
- `SystemMessage` → System prompts
- `UserMessage` → User messages  
- `AssistantMessage` → Assistant responses
- `FunctionExecutionResultMessage` → Tool results

### 2. Token Management
- Automatic token counting
- Usage tracking per request and total
- Context window management

### 3. JSON and Structured Output
- Support for JSON mode
- Pydantic model-based structured output
- Automatic schema instruction injection

### 4. Async Support
- Fully async implementation
- Streaming support
- Proper resource cleanup

## Implementation Notes

### Updating for the Real SDK

When integrating with the actual Claude Code SDK, update these sections in `claude_code_model_client.py`:

1. **Imports** (line ~31):
```python
from claude_code_sdk import ClaudeCodeSDK, Message, MessageType
```

2. **SDK Initialization** (line ~67):
```python
self._sdk = ClaudeCodeSDK(api_key=api_key, **kwargs)
```

3. **API Call** (line ~120):
```python
response = await self._sdk.query(
    messages=sdk_messages,
    **query_params
)
```

4. **Response Parsing** (line ~128):
```python
# Parse actual SDK response
content = response.content
usage = RequestUsage(
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens
)
```

5. **Streaming** (if supported):
```python
async for chunk in self._sdk.stream_query(messages=sdk_messages, **params):
    yield chunk.content
```

### Error Handling

Add proper error handling for SDK-specific exceptions:

```python
from claude_code_sdk import ClaudeCodeError

try:
    response = await self._sdk.query(messages)
except ClaudeCodeError as e:
    # Handle SDK-specific errors
    raise ModelClientError(f"Claude Code SDK error: {e}")
```

### Model Capabilities

Update `model_info` based on actual Claude Code capabilities:

```python
@property
def model_info(self) -> ModelInfo:
    return {
        "vision": True,  # If image support
        "function_calling": True,  # If tool support
        "json_output": True,
        "family": "claude-code",
        "structured_output": True,  # If supported
        "multiple_system_messages": True  # If supported
    }
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure the Claude Code SDK is installed
2. **API Key Error**: Check environment variable is set
3. **Token Limit**: Monitor token usage with `model_client.total_usage()`
4. **Async Errors**: Always use `async`/`await` properly

### Debug Logging

Enable debug logging to see detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Context Management

```python
from autogen_core.model_context import BufferedChatCompletionContext

context = BufferedChatCompletionContext(buffer_size=10)
agent = AssistantAgent(
    "agent",
    model_client=model_client,
    model_context=context
)
```

### Rate Limiting

```python
import asyncio

class RateLimitedClient(ClaudeCodeChatCompletionClient):
    def __init__(self, *args, max_requests_per_minute=60, **kwargs):
        super().__init__(*args, **kwargs)
        self._rate_limiter = asyncio.Semaphore(max_requests_per_minute)
        
    async def create(self, *args, **kwargs):
        async with self._rate_limiter:
            return await super().create(*args, **kwargs)
```

## Contributing

When contributing improvements:
1. Maintain compatibility with AutoGen's `ChatCompletionClient` interface
2. Add appropriate error handling
3. Update documentation
4. Include usage examples

## License

This integration follows AutoGen's licensing terms. Ensure compliance with both AutoGen and Claude Code SDK licenses.