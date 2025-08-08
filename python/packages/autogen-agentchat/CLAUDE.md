# CLAUDE.md - AutoGen AgentChat Package

*I use this guide to help you master the AgentChat API - from your first agent to complex multi-agent systems. Let me guide you through practical examples and best practices.*

This guide helps developers understand how to use the AutoGen AgentChat API for building multi-agent applications.

## Overview

The AgentChat API provides a high-level, opinionated interface for creating multi-agent systems. It's designed for rapid prototyping and implements common patterns like two-agent chats and group conversations.

## Key Components

### 1. Agents (`autogen_agentchat.agents`)

Available agent types for building your applications:

- **AssistantAgent**: LLM-powered agent for general tasks
  - Supports tools, system messages, and model configuration
  - Can handle streaming responses
  - Example: `AssistantAgent("name", model_client, tools=[...])`

- **UserProxyAgent**: Represents human input in conversations
  - Collects input from users during execution
  - Can be configured with input functions
  - Example: `UserProxyAgent("user", input_func=custom_input)`

- **CodeExecutorAgent**: Executes code in sandboxed environments
  - Works with code executors from autogen-ext
  - Handles code execution results
  - Example: `CodeExecutorAgent("executor", code_executor=executor)`

- **SocietyOfMindAgent**: Coordinates multiple inner agents
  - Manages a team of agents internally
  - Delegates tasks to appropriate sub-agents
  - Example: `SocietyOfMindAgent("coordinator", inner_agents=[...])`

### 2. Teams (`autogen_agentchat.teams`)

Multi-agent coordination patterns:

- **RoundRobinGroupChat**: Agents speak in sequential order
  ```python
  team = RoundRobinGroupChat(
      participants=[agent1, agent2, agent3],
      termination_condition=MaxMessageTermination(10)
  )
  ```

- **SelectorGroupChat**: Model dynamically selects next speaker
  ```python
  team = SelectorGroupChat(
      participants=[agent1, agent2],
      model_client=model_client,
      selector_prompt="Select the best agent for: {task}"
  )
  ```

- **SwarmGroupChat**: Agents hand off tasks based on capabilities
  ```python
  team = SwarmGroupChat(
      participants=[agent1, agent2],
      termination_condition=TextMentionTermination("DONE")
  )
  ```

### 3. Messages (`autogen_agentchat.messages`)

Message types for agent communication:

- **TextMessage**: Standard text messages
- **MultiModalMessage**: Messages with text and images
- **ToolCallRequestEvent**: Tool execution requests
- **ToolCallExecutionEvent**: Tool execution results
- **HandoffMessage**: Agent handoff with context
- **StopMessage**: Signals conversation termination

### 4. Conditions (`autogen_agentchat.conditions`)

Termination conditions for controlling conversations:

- **MaxMessageTermination**: Stop after N messages
- **TextMentionTermination**: Stop when specific text appears
- **HandoffTermination**: Stop on agent handoff
- **TimeoutTermination**: Stop after time limit
- **ExternalStopTermination**: Stop on external signal

### 5. UI Components (`autogen_agentchat.ui`)

- **Console**: Terminal-based UI for streaming agent conversations
  ```python
  await Console(team.run_stream(task="Your task"))
  ```

## Common Usage Patterns

### Basic Two-Agent Chat
```python
assistant = AssistantAgent("assistant", model_client)
user = UserProxyAgent("user")

# Single exchange
result = await assistant.run(task="Hello!")

# Continuous conversation
await Console(
    RoundRobinGroupChat([assistant, user]).run_stream(task="Let's chat")
)
```

### Tool-Using Agent
```python
from autogen_core.tools import FunctionTool

def search_web(query: str) -> str:
    return f"Results for {query}"

tool = FunctionTool(search_web)
agent = AssistantAgent("researcher", model_client, tools=[tool])
```

### Code Execution Team
```python
from autogen_ext.code_executors import LocalCommandLineCodeExecutor

executor = LocalCommandLineCodeExecutor()
code_agent = CodeExecutorAgent("coder", code_executor=executor)
assistant = AssistantAgent("helper", model_client)

team = RoundRobinGroupChat([assistant, code_agent])
```

### Custom Termination
```python
# Combine multiple conditions
from autogen_agentchat.conditions import StopMessageTermination

termination = MaxMessageTermination(20) | TextMentionTermination("STOP")
team = RoundRobinGroupChat(agents, termination_condition=termination)
```

## State Management

AgentChat supports saving and loading conversation state:

```python
# Save state
state = await team.save_state()

# Load state
await team.load_state(state)
```

## Streaming and Events

Handle real-time updates during execution:

```python
async for message in team.run_stream(task="Your task"):
    if isinstance(message, TextMessage):
        print(f"{message.source}: {message.content}")
    elif isinstance(message, ToolCallRequestEvent):
        print(f"Calling tool: {message.tool_call.name}")
```

## Best Practices

1. **Choose the right team type**:
   - RoundRobin for predictable turn-taking
   - Selector for dynamic conversations
   - Swarm for capability-based handoffs

2. **Set appropriate termination conditions** to prevent infinite loops

3. **Use streaming** for better user experience with long tasks

4. **Handle errors gracefully**:
   ```python
   try:
       result = await team.run(task)
   except Exception as e:
       print(f"Team execution failed: {e}")
   ```

5. **Configure agents with clear roles** using system messages

## Integration with Core and Extensions

- Use model clients from `autogen-ext.models`
- Add code execution with `autogen-ext.code_executors`
- Enhance with tools from `autogen-ext.tools`
- Build on core runtime from `autogen-core` for advanced scenarios

Remember: AgentChat is designed for rapid prototyping. For production systems requiring fine control, consider using the Core API directly.