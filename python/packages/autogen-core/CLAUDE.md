# CLAUDE.md - AutoGen Core Package

*When you need maximum control and flexibility, I'll guide you through Core's powerful low-level APIs. Think of me as your expert companion for advanced AutoGen development.*

This guide helps developers understand the foundational AutoGen Core API for building distributed, event-driven multi-agent systems.

## Overview

AutoGen Core provides the low-level building blocks for agent systems: message passing, event-driven agents, runtimes, and cross-language support. It's designed for flexibility and control when building production systems.

## Key Concepts

### 1. Agents and Message Passing

Agents in Core are event-driven and communicate through typed messages:

```python
from autogen_core import Agent, MessageContext, TopicId

class MyAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
    
    async def on_message(self, message: Any, ctx: MessageContext) -> Any:
        # Process incoming message
        if isinstance(message, TextMessage):
            # Send response to another agent
            await self.send_message(
                RecipientAgent, 
                ResponseMessage(content="Received: " + message.content),
                topic_id=TopicId("response_topic")
            )
```

### 2. Runtime System

The runtime manages agent lifecycle and message routing:

```python
from autogen_core import SingleThreadedAgentRuntime

# Create runtime
runtime = SingleThreadedAgentRuntime()

# Register agents
await runtime.register(
    "my_agent",
    lambda: MyAgent("agent1")
)

# Start runtime
await runtime.start()

# Send messages
await runtime.send_message(
    "my_agent",
    TextMessage(content="Hello"),
    TopicId("default")
)
```

### 3. Topics and Subscriptions

Control message routing with topics and subscriptions:

```python
from autogen_core import TypeSubscription, DefaultTopicId

# Subscribe to specific message types
@default_subscription
class ProcessingAgent(Agent):
    def __init__(self):
        super().__init__("processor")
    
    # Automatically receives all TextMessage instances
    async def on_text_message(self, message: TextMessage, ctx: MessageContext):
        print(f"Processing: {message.content}")

# Or use explicit subscriptions
runtime.add_subscription(
    TypeSubscription("processing_topic", TextMessage)
)
```

### 4. Model Context

Manage conversation context for LLM interactions:

```python
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage, AssistantMessage

# Buffered context (keeps last N messages)
context = BufferedChatCompletionContext(buffer_size=10)

# Add messages
await context.add_message(UserMessage(content="Hello", source="user"))
await context.add_message(AssistantMessage(content="Hi!", source="assistant"))

# Get messages for model
messages = await context.get_messages()
```

### 5. Tools System

Define tools that agents can use:

```python
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, Field

class WeatherArgs(BaseModel):
    location: str = Field(description="City name")
    units: str = Field(default="celsius", pattern="celsius|fahrenheit")

async def get_weather(location: str, units: str) -> str:
    # Implementation
    return f"Weather in {location}: 22°{units[0].upper()}"

# Create tool
weather_tool = FunctionTool(
    get_weather,
    name="get_weather",
    description="Get current weather for a location"
)
```

### 6. Memory Interface

Implement memory systems for agents:

```python
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult

class VectorMemory(Memory):
    async def add(self, content: MemoryContent) -> None:
        # Store in vector database
        pass
    
    async def query(self, query: str) -> MemoryQueryResult:
        # Semantic search
        results = await self._search(query)
        return MemoryQueryResult(results=results)
    
    async def clear(self) -> None:
        # Clear storage
        pass
```

## Agent Patterns

### Basic Agent
```python
from autogen_core import Agent, message_handler

class CalculatorAgent(Agent):
    @message_handler
    async def handle_calculation(self, msg: CalculateRequest) -> CalculateResponse:
        result = eval(msg.expression)  # Simple example
        return CalculateResponse(result=result)
```

### Routed Agent
```python
from autogen_core import RoutedAgent, route

class MultiPurposeAgent(RoutedAgent):
    @route(MessageType.QUERY)
    async def handle_query(self, msg: QueryMessage) -> Response:
        # Handle queries
        pass
    
    @route(MessageType.COMMAND)
    async def handle_command(self, msg: CommandMessage) -> None:
        # Execute commands
        pass
```

### Stateful Agent
```python
class StatefulAgent(Agent):
    def __init__(self):
        super().__init__("stateful")
        self._state = {}
    
    async def on_message(self, message: Any, ctx: MessageContext):
        if isinstance(message, StoreMessage):
            self._state[message.key] = message.value
        elif isinstance(message, RetrieveMessage):
            return self._state.get(message.key)
```

## Components and Configuration

### Component System
```python
from autogen_core import Component, ComponentModel
from pydantic import BaseModel

class MyComponentConfig(BaseModel):
    setting1: str
    setting2: int = 10

class MyComponent(Component[MyComponentConfig]):
    component_type = "my_component"
    component_config_schema = MyComponentConfig
    
    def __init__(self, setting1: str, setting2: int = 10):
        self._config = MyComponentConfig(setting1=setting1, setting2=setting2)
    
    def _to_config(self) -> MyComponentConfig:
        return self._config
    
    @classmethod
    def _from_config(cls, config: MyComponentConfig) -> "MyComponent":
        return cls(config.setting1, config.setting2)
```

### Serialization
```python
# Save component configuration
component = MyComponent("value", 20)
config_dict = component.dump_component()

# Load from configuration
loaded = Component.load_component(config_dict)
```

## Distributed Runtime

Run agents across multiple processes or machines:

```python
from autogen_core import WorkerAgentRuntime

# Worker node
worker = WorkerAgentRuntime(host="0.0.0.0", port=50051)
worker.register("worker_agent", WorkerAgent)
await worker.start()

# Coordinator node
runtime = DistributedRuntime()
await runtime.add_worker("worker1", "localhost:50051")
```

## Cancellation and Timeouts

Handle long-running operations:

```python
from autogen_core import CancellationToken
import asyncio

class TimeoutAgent(Agent):
    async def on_message(self, message: Any, ctx: MessageContext):
        token = CancellationToken()
        
        # Set timeout
        asyncio.create_task(self._timeout(token, 30))
        
        try:
            result = await self._long_operation(token)
            return result
        except asyncio.CancelledError:
            return ErrorResponse("Operation timed out")
    
    async def _timeout(self, token: CancellationToken, seconds: float):
        await asyncio.sleep(seconds)
        token.cancel()
```

## Error Handling

```python
from autogen_core.exceptions import AgentError

class RobustAgent(Agent):
    async def on_message(self, message: Any, ctx: MessageContext):
        try:
            return await self._process(message)
        except AgentError as e:
            # Log and handle agent-specific errors
            logger.error(f"Agent error: {e}")
            return ErrorMessage(str(e))
        except Exception as e:
            # Handle unexpected errors
            logger.exception("Unexpected error")
            return ErrorMessage("Internal error occurred")
```

## Testing Agents

```python
import pytest
from autogen_core import InMemoryRuntime

@pytest.mark.asyncio
async def test_my_agent():
    # Use in-memory runtime for testing
    runtime = InMemoryRuntime()
    
    # Register agent
    agent = MyAgent()
    runtime.register_agent(agent)
    
    # Send test message
    response = await runtime.send_message(
        agent.id,
        TestMessage(content="test"),
        TopicId("test")
    )
    
    assert response.success
```

## Performance Optimization

### Message Batching
```python
class BatchProcessor(Agent):
    def __init__(self):
        super().__init__("batch_processor")
        self._buffer = []
        self._flush_task = None
    
    async def on_message(self, message: Any, ctx: MessageContext):
        self._buffer.append(message)
        
        if len(self._buffer) >= 100:
            await self._flush()
        elif not self._flush_task:
            self._flush_task = asyncio.create_task(self._delayed_flush())
    
    async def _flush(self):
        # Process batch
        await self._process_batch(self._buffer)
        self._buffer.clear()
```

### Connection Pooling
```python
class PooledAgent(Agent):
    def __init__(self, pool_size: int = 10):
        super().__init__("pooled")
        self._pool = AsyncConnectionPool(size=pool_size)
    
    async def on_message(self, message: Any, ctx: MessageContext):
        async with self._pool.acquire() as conn:
            return await conn.execute(message)
```

## Integration with AgentChat

Core agents can work with AgentChat's high-level API:

```python
from autogen_core import Agent as CoreAgent
from autogen_agentchat.base import BaseChatAgent

class HybridAgent(BaseChatAgent):
    def __init__(self, core_agent: CoreAgent):
        self._core_agent = core_agent
    
    async def on_messages(self, messages, ctx):
        # Delegate to core agent
        return await self._core_agent.on_message(messages[0], ctx)
```

## Best Practices

1. **Design for asynchrony** - All agent operations should be async
2. **Use typed messages** - Define clear message schemas with Pydantic
3. **Handle errors gracefully** - Don't let one agent crash the system
4. **Implement proper cleanup** - Use `__aexit__` for resource management
5. **Test with in-memory runtime** - Faster and more predictable for tests
6. **Use topics for organization** - Group related messages by topic
7. **Consider message size** - Large messages can impact performance
8. **Implement health checks** - Monitor agent status in production

## Additional Resources

### Documentation
For comprehensive documentation and guides, see:
- **Full Documentation Hub**: `docs/CLAUDE.md` - Complete navigation guide for all AutoGen documentation including tutorials, API references, and examples

Remember: Core is for when you need maximum control and flexibility. For rapid prototyping, consider using AgentChat instead.