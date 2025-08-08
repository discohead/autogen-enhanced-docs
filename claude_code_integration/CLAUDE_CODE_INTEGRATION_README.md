# Claude Code SDK Integration for AutoGen

This guide shows two ways to integrate Claude Code SDK with AutoGen:
1. **ClaudeCodeChatCompletionClient**: A model client for simple LLM usage
2. **ClaudeCodeChatAgent**: A full agent implementation that preserves Claude Code's native tool execution

## Overview

### ClaudeCodeChatCompletionClient
A custom `ChatCompletionClient` implementation that:
- Uses the Claude Code SDK to query your local Claude Code installation
- Maps between AutoGen and Claude Code message formats
- Handles streaming responses
- Provides full async support
- **Limitation**: Does not properly handle Claude Code's internal tool execution

### ClaudeCodeChatAgent (Recommended for Tool Usage)
A custom `BaseChatAgent` implementation that:
- Preserves Claude Code's native tool execution (Read, Write, Edit, etc.)
- Emits proper events for tool usage visibility
- Supports handoffs for team coordination
- Works seamlessly in AutoGen teams
- Maintains conversation state

## Prerequisites

1. **Install Claude Code SDK:**
   ```bash
   pip install claude-code-sdk
   ```

2. **Install Claude Code CLI:**
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

3. **Install AutoGen:**
   ```bash
   pip install autogen-agentchat autogen-ext[openai]
   ```

## Quick Start

### Using ClaudeCodeChatAgent (Recommended for Tool Usage)

```python
import asyncio
from claude_code_chat_agent import ClaudeCodeChatAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

async def main():
    # Create Claude Code agent with tools
    agent = ClaudeCodeChatAgent(
        name="claude_coder",
        description="A helpful coding assistant with file access",
        system_prompt="You are an expert Python developer.",
        allowed_tools=["Read", "Write", "Edit"],  # Claude Code tools
        permission_mode="acceptEdits",  # Auto-accept file edits
    )
    
    # Run task
    result = await agent.run(task="Create a hello.py file with a greeting function")
    print(result.messages[-1].content)
    
    # Or use in a team with streaming
    await Console(agent.run_stream(task="Explain the code you just wrote"))

asyncio.run(main())
```

### Using ClaudeCodeChatCompletionClient (Simple LLM Usage)

```python
import asyncio
from claude_code_model_client import ClaudeCodeChatCompletionClient
from autogen_agentchat.agents import AssistantAgent

async def main():
    # Create Claude Code model client
    model_client = ClaudeCodeChatCompletionClient(
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a helpful AI assistant.",
        # Note: Tools configured here won't work properly with AutoGen
    )
    
    # Create agent with Claude Code
    agent = AssistantAgent(
        name="claude_assistant",
        model_client=model_client,
    )
    
    # Run task
    result = await agent.run(task="Hello! What can you help me with?")
    print(result.messages[-1].content)
    
    # Cleanup
    await model_client.close()

asyncio.run(main())
```

## Configuration Options

### ClaudeCodeChatAgent Parameters

- **`name`** (str): Unique agent name within a team
- **`description`** (str): Agent capabilities description
- **`system_prompt`** (str): System instructions for Claude
- **`allowed_tools`** (List[str]): Claude Code tools to enable (e.g., ["Read", "Write", "Edit", "Bash"])
- **`permission_mode`** (str): Permission mode for tools:
  - `"default"`: CLI prompts for dangerous operations
  - `"acceptEdits"`: Auto-accept file edits
  - `"bypassPermissions"`: Allow all operations (use with caution)
- **`max_turns`** (int): Maximum conversation turns
- **`cwd`** (str): Working directory for Claude Code operations
- **`model`** (str): Model name (default: "claude-3-5-sonnet-20241022")
- **`handoffs`** (List): Agents this agent can hand off to
- **`emit_tool_events`** (bool): Whether to emit events for tool usage (default: True)

### ClaudeCodeChatCompletionClient Parameters

- **`model`** (str): Model name to use (default: "claude-3-5-sonnet-20241022")
- **`system_prompt`** (str): Default system prompt for conversations
- **`allowed_tools`** (List[str]): Claude Code tools to enable (e.g., ["Read", "Write", "Edit", "Bash"])
- **`permission_mode`** (str): Permission mode for tools:
  - `"default"`: CLI prompts for dangerous operations
  - `"acceptEdits"`: Auto-accept file edits
  - `"bypassPermissions"`: Allow all operations (use with caution)
- **`max_turns`** (int): Maximum conversation turns
- **`cwd`** (str): Working directory for Claude Code operations
- **Additional options**: Any other options supported by `ClaudeCodeOptions`

## Using Claude Code Built-in Tools

Claude Code comes with powerful built-in tools that can be enabled:

```python
model_client = ClaudeCodeChatCompletionClient(
    allowed_tools=[
        "Read",      # Read files
        "Write",     # Write files
        "Edit",      # Edit files
        "Bash",      # Run bash commands
        "Grep",      # Search files
        "LS",        # List directory contents
        # See Claude Code docs for full list
    ],
    permission_mode="acceptEdits",  # Auto-accept file operations
)
```

## Multi-Agent Teams

### Using ClaudeCodeChatAgent in Teams

Create teams where Claude Code agents collaborate with full tool support:

```python
from claude_code_chat_agent import ClaudeCodeChatAgent
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.code_executors import LocalCommandLineCodeExecutor

# Create Claude Code agent with file tools
claude_coder = ClaudeCodeChatAgent(
    name="coder",
    description="Writes and modifies code files",
    allowed_tools=["Read", "Write", "Edit"],
    permission_mode="acceptEdits",
    system_prompt="You are an expert Python developer."
)

# Create code executor agent
executor = CodeExecutorAgent(
    name="executor",
    code_executor=LocalCommandLineCodeExecutor(work_dir="./workspace")
)

# Create team
team = RoundRobinGroupChat(
    participants=[claude_coder, executor],
    termination_condition=MaxMessageTermination(max_messages=10)
)

# Run collaborative task
result = await team.run(task="Write a fibonacci.py script and test it")
```

### Swarm Teams with Handoffs

```python
from autogen_agentchat.teams import SwarmGroupChat
from autogen_agentchat.conditions import HandoffTermination

# Create specialized agents with handoff capabilities
architect = ClaudeCodeChatAgent(
    name="architect",
    description="Designs software architecture",
    system_prompt="You design software systems. Hand off to 'implementer' for coding.",
    handoffs=["implementer", "reviewer"]
)

implementer = ClaudeCodeChatAgent(
    name="implementer",
    description="Implements code based on designs",
    allowed_tools=["Read", "Write", "Edit"],
    system_prompt="You implement code. Hand off to 'reviewer' when done.",
    handoffs=["reviewer", "architect"]
)

reviewer = ClaudeCodeChatAgent(
    name="reviewer",
    description="Reviews code quality",
    allowed_tools=["Read"],
    system_prompt="You review code. Hand off to 'implementer' for fixes.",
    handoffs=["implementer", "TERMINATE"]
)

# Create swarm team
swarm = SwarmGroupChat(
    participants=[architect, implementer, reviewer],
    termination_condition=HandoffTermination(target="TERMINATE")
)

# Run with automatic handoffs
result = await swarm.run(task="Design and implement a REST API client")
```

## Streaming Responses

### Streaming with ClaudeCodeChatAgent

See tool execution in real-time:

```python
from autogen_agentchat.ui import Console

agent = ClaudeCodeChatAgent(
    name="claude",
    allowed_tools=["Read", "Write"],
    emit_tool_events=True  # See tool usage events
)

# Stream to console with tool visibility
await Console(agent.run_stream(task="Analyze the project structure and create a summary"))

# Or handle events manually
async for event in agent.run_stream(task="Create a test file"):
    if isinstance(event, ToolCallRequestEvent):
        print(f"🔧 Using tool: {event.content[0].name}")
    elif isinstance(event, ToolCallExecutionEvent):
        print(f"✅ Tool completed: {event.content[0].call_id}")
    elif isinstance(event, TextMessage):
        print(f"💬 {event.source}: {event.content}")
```

### Streaming with Model Client

For simple LLM streaming without tools:

```python
from autogen_agentchat.ui import Console

agent = AssistantAgent(
    "claude",
    model_client=ClaudeCodeChatCompletionClient(),
    model_client_stream=True,
)

await Console(agent.run_stream(task="Tell me a story"))
```

## Custom AutoGen Tools

You can combine Claude Code with custom AutoGen tools:

```python
from autogen_core.tools import FunctionTool

def my_custom_tool(param: str) -> str:
    """My custom tool function."""
    return f"Processed: {param}"

agent = AssistantAgent(
    "claude_with_tools",
    model_client=model_client,
    tools=[FunctionTool(my_custom_tool)],  # AutoGen tools
)
```

## Advanced Usage

### Working Directory

Set a specific working directory for Claude Code operations:

```python
model_client = ClaudeCodeChatCompletionClient(
    cwd="/path/to/project"
)
```

### Session Continuation

Continue previous Claude Code sessions:

```python
model_client = ClaudeCodeChatCompletionClient(
    continue_conversation=True,
    resume="session-id"  # From previous session
)
```

### Error Handling

Handle Claude Code specific errors:

```python
from claude_code_sdk import CLINotFoundError, ProcessError

try:
    result = await agent.run(task="...")
except CLINotFoundError:
    print("Please install Claude Code CLI")
except ProcessError as e:
    print(f"Claude Code error: {e}")
```

## Choosing Between ClaudeCodeChatAgent and Model Client

### Use ClaudeCodeChatAgent when:
- You need Claude Code's built-in tools (Read, Write, Edit, Bash, etc.)
- You want to see tool execution events
- You need handoffs between agents
- You're building multi-agent teams
- You want full integration with AutoGen's agent ecosystem

### Use ClaudeCodeChatCompletionClient when:
- You only need LLM text generation without tools
- You're using existing AssistantAgent code
- You want to mix Claude Code with AutoGen's function tools (though this has limitations)
- You need a drop-in replacement for other model clients

## Limitations

### ClaudeCodeChatCompletionClient Limitations:
1. **Tool Execution Conflict**: Cannot properly distinguish between Claude Code's internal tools and AutoGen's external functions
2. **Lost Tool Context**: Tool execution results from Claude Code are not properly handled
3. **No Bidirectional Communication**: Cannot send function results back to Claude in the proper format

### ClaudeCodeChatAgent Limitations:
1. **No Custom AutoGen Tools**: Cannot use AutoGen's FunctionTool with this agent
2. **Tool Set Fixed**: Tools are Claude Code's built-in set only
3. **State Serialization**: Complex state restoration needs proper message deserialization

### General Limitations:
1. **Token Tracking**: Token usage is tracked per request, not cumulatively
2. **Cost Tracking**: Cost information comes from Claude Code's response
3. **Session Management**: Continue/resume functionality requires session ID tracking

## Examples

See `example_claude_code_agent.py` for complete examples including:
- Basic usage
- File operations with tools
- Multi-agent teams
- Streaming responses
- Custom tool integration

## Troubleshooting

1. **Claude Code not found**: Ensure Claude Code CLI is installed and in PATH
2. **Permission errors**: Check `permission_mode` setting
3. **Tool errors**: Verify tools are in `allowed_tools` list
4. **Async errors**: Remember to use `async`/`await` properly

## Benefits

- **Local Execution**: No API calls, everything runs locally
- **Full Tool Access**: Use all Claude Code's powerful built-in tools
- **AutoGen Integration**: Works seamlessly with all AutoGen features
- **Cost Effective**: No API costs, just local compute
- **Privacy**: Data stays on your machine