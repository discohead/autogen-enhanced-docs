# CLAUDE.md - AutoGen AgentChat Agents

*As your AutoGen mentor, I'll help you understand each agent type's strengths and guide you in choosing the right agents for your specific use case.*

This guide provides detailed information about the agent types available in AutoGen AgentChat for building multi-agent applications.

## Agent Types Overview

All agents in AgentChat inherit from `BaseChatAgent` and implement the high-level chat protocol. Each agent type is designed for specific use cases.

## AssistantAgent

The most versatile agent type, powered by LLMs with optional tool use.

### Basic Usage
```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
    model_client_stream=True  # Enable streaming
)
```

### Advanced Features

#### Tool Usage
```python
from autogen_core.tools import FunctionTool

async def search_web(query: str) -> str:
    # Implementation
    return f"Results for {query}"

tool = FunctionTool(search_web, description="Search the web")
agent = AssistantAgent(
    "researcher",
    model_client=model_client,
    tools=[tool],
    tool_call_summary_format="{tool_name} returned: {result}"
)
```

#### Handoffs
```python
from autogen_agentchat.base import Handoff

agent = AssistantAgent(
    "coordinator",
    model_client=model_client,
    handoffs=[
        Handoff(target="specialist", message="Hand off to specialist for technical questions"),
        Handoff(target="writer", message="Hand off to writer for content creation")
    ]
)
```

#### Model Context Management
```python
from autogen_core.model_context import BufferedChatCompletionContext

# Keep only last 10 messages in context
context = BufferedChatCompletionContext(buffer_size=10)
agent = AssistantAgent(
    "assistant",
    model_client=model_client,
    model_context=context
)
```

#### Reflection
```python
agent = AssistantAgent(
    "thinker",
    model_client=model_client,
    reflect_on_tool_use=True,
    reflection_prompt="Analyze if the tool results answer the user's question completely."
)
```

## UserProxyAgent

Represents human input in multi-agent conversations.

### Basic Usage
```python
from autogen_agentchat.agents import UserProxyAgent

# Simple user proxy
user = UserProxyAgent("user")

# With custom input function
async def get_user_input(prompt: str) -> str:
    # Custom input logic
    return input(f"[USER] {prompt}: ")

user = UserProxyAgent(
    "user",
    input_func=get_user_input
)
```

### Silent Mode
```python
# No input requested - useful for testing
user = UserProxyAgent(
    "user",
    input_func=lambda p: "Default response"
)
```

## CodeExecutorAgent

Executes code safely using various execution backends.

### Local Execution
```python
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.code_executors import LocalCommandLineCodeExecutor

executor = LocalCommandLineCodeExecutor(
    work_dir="./workspace",
    timeout=30
)

code_agent = CodeExecutorAgent(
    "python_executor",
    code_executor=executor,
    execute_on_reply=True  # Auto-execute code blocks in messages
)
```

### Docker Execution
```python
from autogen_ext.code_executors import DockerCommandLineCodeExecutor

docker_executor = DockerCommandLineCodeExecutor(
    image="python:3.11",
    work_dir="/workspace"
)

secure_code_agent = CodeExecutorAgent(
    "secure_executor",
    code_executor=docker_executor
)
```

### Virtual Environment
```python
from autogen_ext.code_executors import VirtualEnvContext

venv_executor = LocalCommandLineCodeExecutor(
    virtual_env_context=VirtualEnvContext(
        work_dir="./venv",
        pip_requirements=["numpy", "pandas", "matplotlib"]
    )
)

data_agent = CodeExecutorAgent(
    "data_analyst",
    code_executor=venv_executor
)
```

## SocietyOfMindAgent

Manages a team of inner agents to handle complex tasks.

### Basic Usage
```python
from autogen_agentchat.agents import SocietyOfMindAgent
from autogen_agentchat.teams import RoundRobinGroupChat

# Create inner agents
researcher = AssistantAgent("researcher", model_client, 
    system_message="You research information")
writer = AssistantAgent("writer", model_client,
    system_message="You write content")

# Create inner team
inner_team = RoundRobinGroupChat([researcher, writer])

# Create SocietyOfMind agent
som_agent = SocietyOfMindAgent(
    "content_creator",
    team=inner_team,
    model_client=model_client
)
```

### Custom Task Preparation
```python
som_agent = SocietyOfMindAgent(
    "project_manager",
    team=inner_team,
    model_client=model_client,
    instruction="Break down tasks and delegate to team members",
    max_inner_messages=50
)
```

## Common Patterns

### Agent with Memory
```python
from autogen_core.memory import ListMemory

memory = ListMemory()
agent = AssistantAgent(
    "memorable",
    model_client=model_client,
    memory=memory
)

# Memory is automatically queried and updated during conversations
```

### Agent with Custom System Message
```python
agent = AssistantAgent(
    "specialist",
    model_client=model_client,
    system_message="""You are a Python expert. When answering:
    1. Provide working code examples
    2. Explain the logic clearly
    3. Suggest best practices
    4. Mention common pitfalls"""
)
```

### Tool-Executing Agent Pattern
```python
# Combine AssistantAgent with tools and CodeExecutorAgent
planner = AssistantAgent(
    "planner",
    model_client=model_client,
    system_message="Write Python code to solve problems"
)

executor = CodeExecutorAgent(
    "executor",
    code_executor=LocalCommandLineCodeExecutor()
)

# Use in a team
team = RoundRobinGroupChat([planner, executor])
```

### Multi-Model Agent Setup
```python
# Different models for different agents
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

gpt4_agent = AssistantAgent(
    "precise",
    model_client=OpenAIChatCompletionClient(model="gpt-4"),
    system_message="Provide accurate, detailed analysis"
)

claude_agent = AssistantAgent(
    "creative",
    model_client=AnthropicChatCompletionClient(model="claude-3-5-sonnet"),
    system_message="Generate creative solutions"
)
```

## Agent Lifecycle

### Initialization
```python
# Agents can be initialized with configuration
agent = AssistantAgent(
    name="configured_agent",
    model_client=model_client,
    tools=[tool1, tool2],
    handoffs=[handoff1],
    system_message="Custom instructions",
    description="Handles specific tasks"  # Used by team managers
)
```

### State Management
```python
# Save agent state
state = await agent.save_state()

# Restore agent state
await agent.load_state(state)
```

### Cleanup
```python
# Agents don't typically need explicit cleanup
# But model clients should be closed
await model_client.close()
```

## Error Handling

```python
from autogen_agentchat.base import TaskResult

try:
    result = await agent.run(task="Complex task")
    if result.stop_reason == "error":
        print(f"Task failed: {result.messages[-1].content}")
except Exception as e:
    print(f"Agent execution error: {e}")
```

## Performance Considerations

1. **Model Client Reuse**: Share model clients between agents
2. **Context Size**: Use BufferedChatCompletionContext for long conversations
3. **Streaming**: Enable for better perceived performance
4. **Tool Timeout**: Set appropriate timeouts for tools
5. **Code Execution**: Use Docker for untrusted code

## Testing Agents

```python
from autogen_ext.models.replay import ReplayChatCompletionClient

# Deterministic testing
test_client = ReplayChatCompletionClient(
    responses=["Response 1", "Response 2"]
)

test_agent = AssistantAgent("test", model_client=test_client)
result = await test_agent.run(task="Test task")
assert "Response 1" in result.messages[-1].content
```

Remember: Choose the right agent type for your use case. AssistantAgent for general AI tasks, UserProxyAgent for human interaction, CodeExecutorAgent for code execution, and SocietyOfMindAgent for complex multi-step tasks.