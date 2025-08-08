# CLAUDE.md - AutoGen Troubleshooting Guide

*As your AutoGen mentor, I'll help you diagnose and fix common issues. This guide contains solutions I've gathered from helping many users overcome similar challenges.*

This comprehensive troubleshooting guide helps you resolve common issues when building with AutoGen.

## Quick Diagnostic Checklist

Before diving into specific issues, run through this checklist:

```python
# 1. Check AutoGen version
import autogen_agentchat
import autogen_core
import autogen_ext
print(f"AgentChat: {autogen_agentchat.__version__}")
print(f"Core: {autogen_core.__version__}")
print(f"Extensions: {autogen_ext.__version__}")

# 2. Verify model client connection
from autogen_ext.models.openai import OpenAIChatCompletionClient
try:
    client = OpenAIChatCompletionClient(model="gpt-4o")
    response = await client.create([{"role": "user", "content": "test"}])
    print("✓ Model client working")
except Exception as e:
    print(f"✗ Model client error: {e}")

# 3. Check async environment
import asyncio
print(f"✓ Async available: {asyncio.get_event_loop() is not None}")
```

## Common Issues and Solutions

### 1. Import Errors

#### Problem: "No module named 'autogen_agentchat'"
```python
# Error
ImportError: No module named 'autogen_agentchat'
```

**Solution**:
```bash
# Install the correct package
pip install autogen-agentchat

# Not 'autogen' or 'pyautogen' - those are old versions
```

#### Problem: Missing extensions
```python
# Error
ImportError: cannot import name 'OpenAIChatCompletionClient' from 'autogen_ext.models'
```

**Solution**:
```bash
# Install with specific extensions
pip install "autogen-ext[openai]"

# For multiple extensions
pip install "autogen-ext[openai,anthropic,docker]"
```

### 2. Async/Await Issues

#### Problem: "RuntimeWarning: coroutine was never awaited"
```python
# Wrong
result = agent.run(task="Hello")  # Missing await

# Correct
result = await agent.run(task="Hello")
```

**Solution for Jupyter/Notebooks**:
```python
# Option 1: Use await directly (Jupyter supports top-level await)
result = await agent.run(task="Hello")

# Option 2: Use asyncio.run()
import asyncio
result = asyncio.run(agent.run(task="Hello"))

# Option 3: Create event loop
loop = asyncio.get_event_loop()
result = loop.run_until_complete(agent.run(task="Hello"))
```

**Solution for Scripts**:
```python
import asyncio

async def main():
    # Your async code here
    agent = AssistantAgent(...)
    result = await agent.run(task="Hello")
    return result

# Run the main function
if __name__ == "__main__":
    result = asyncio.run(main())
```

### 3. Model Client Errors

#### Problem: "API key not found"
```python
# Error
openai.AuthenticationError: No API key provided
```

**Solutions**:
```python
# Option 1: Environment variable (recommended)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# Option 2: Direct in client
client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key="sk-..."  # Not recommended for production
)

# Option 3: .env file
# Create .env file with:
# OPENAI_API_KEY=sk-...
from dotenv import load_dotenv
load_dotenv()
```

#### Problem: "Model not found"
```python
# Error
openai.NotFoundError: Model 'gpt-4' not found
```

**Solution**:
```python
# Check available models for your API key
# Use models you have access to
client = OpenAIChatCompletionClient(
    model="gpt-3.5-turbo"  # If you don't have GPT-4 access
)
```

### 4. Agent Communication Issues

#### Problem: Agents not responding
```python
# Common mistake - no termination condition
team = RoundRobinGroupChat([agent1, agent2])
# This could run forever!
```

**Solution**:
```python
# Always set termination conditions
from autogen_agentchat.conditions import MaxMessageTermination

team = RoundRobinGroupChat(
    participants=[agent1, agent2],
    termination_condition=MaxMessageTermination(10)
)
```

#### Problem: Empty responses
```python
# Agent returns empty messages
```

**Solution**:
```python
# Check system message
agent = AssistantAgent(
    "assistant",
    model_client=model_client,
    system_message="You are a helpful assistant. Always provide detailed responses."
)

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 5. Tool Execution Failures

#### Problem: "Tool not found"
```python
# Error
ValueError: Unknown tool: 'my_function'
```

**Solution**:
```python
# Ensure proper tool creation
from autogen_core.tools import FunctionTool

def my_function(param: str) -> str:
    """Function description."""  # Required!
    return f"Result: {param}"

# Create tool correctly
tool = FunctionTool(
    my_function,
    name="my_function",  # Must match function name
    description="Clear description"
)

agent = AssistantAgent(
    "agent",
    model_client=model_client,
    tools=[tool]  # Add to agent
)
```

#### Problem: Tool schema errors
```python
# Error
pydantic.ValidationError: Invalid schema
```

**Solution**:
```python
# Use proper type hints
from typing import List, Dict, Optional

def search(
    query: str,  # Simple types
    max_results: int = 5,  # Default values
    filters: Optional[List[str]] = None  # Optional params
) -> List[Dict[str, str]]:
    """Search with filters.
    
    Args:
        query: Search query
        max_results: Maximum results to return
        filters: Optional list of filters
    """
    # Implementation
    return []
```

### 6. Memory and Context Issues

#### Problem: "Context length exceeded"
```python
# Error
openai.BadRequestError: Maximum context length is 4096 tokens
```

**Solutions**:
```python
# Option 1: Use model with larger context
client = OpenAIChatCompletionClient(
    model="gpt-4-turbo",  # 128k context
)

# Option 2: Limit conversation history
from autogen_core.model_context import BufferedChatCompletionContext

context = BufferedChatCompletionContext(
    buffer_size=10  # Keep only last 10 messages
)

agent = AssistantAgent(
    "agent",
    model_client=model_client,
    model_context=context
)

# Option 3: Summarize long conversations
async def summarize_context(messages):
    if len(messages) > 20:
        # Summarize older messages
        summary = await summarizer_agent.run(
            task=f"Summarize: {messages[:10]}"
        )
        return [summary] + messages[-10:]
    return messages
```

### 7. Code Execution Problems

#### Problem: "Docker not found"
```python
# Error
docker.errors.DockerException: Docker not installed
```

**Solutions**:
```bash
# Install Docker
# macOS: brew install docker
# Ubuntu: sudo apt-get install docker.io
# Windows: Download Docker Desktop

# Start Docker service
sudo systemctl start docker  # Linux
open -a Docker  # macOS
```

**Alternative - Use Local Executor**:
```python
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

# Use local execution instead
executor = LocalCommandLineCodeExecutor(
    work_dir="./workspace"
)
```

#### Problem: Code executor timeout
```python
# Error
TimeoutError: Code execution timed out
```

**Solution**:
```python
# Increase timeout
executor = DockerCodeExecutor(
    image="python:3.11",
    timeout=120  # 2 minutes instead of default 30s
)

# Or handle in code
result = await executor.execute_code(
    code="""
import signal
def timeout_handler(signum, frame):
    print("Taking too long, stopping...")
    exit(0)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second limit

# Your long-running code here
""",
    language="python"
)
```

### 8. Team Coordination Issues

#### Problem: Wrong agent responding
```python
# Selector choosing inappropriate agents
```

**Solution**:
```python
# Improve selector prompt
team = SelectorGroupChat(
    participants=[web_agent, code_agent, analyst_agent],
    model_client=model_client,
    selector_prompt="""
    Select the most appropriate agent:
    - web_agent: For web searches and browsing
    - code_agent: For writing and executing code  
    - analyst_agent: For data analysis and insights
    
    Task: {task}
    
    Select one agent name exactly as shown above.
    """
)

# Or use explicit handoffs
web_agent = AssistantAgent(
    "web_agent",
    model_client=model_client,
    handoffs=[
        Handoff(
            target="analyst_agent",
            description="Hand off when analysis is needed"
        )
    ]
)
```

### 9. Streaming Issues

#### Problem: No streaming output
```python
# Not seeing incremental output
```

**Solution**:
```python
# Enable streaming in model client
agent = AssistantAgent(
    "assistant",
    model_client=model_client,
    model_client_stream=True  # Enable streaming
)

# Use streaming UI
from autogen_agentchat.ui import Console

# This will show output as it's generated
await Console(
    agent.run_stream(task="Write a long story")
)

# Or handle manually
async for message in agent.run_stream(task="Task"):
    if isinstance(message, str):
        print(message, end="", flush=True)
```

### 10. Performance Issues

#### Problem: Slow agent responses
```python
# Agents taking too long to respond
```

**Solutions**:
```python
# 1. Use faster models
client = OpenAIChatCompletionClient(
    model="gpt-3.5-turbo"  # Faster than GPT-4
)

# 2. Parallelize independent tasks
async def parallel_research(topics):
    tasks = []
    for topic in topics:
        task = agent.run(task=f"Research {topic}")
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

# 3. Cache repeated queries
from functools import lru_cache

@lru_cache(maxsize=100)
async def cached_agent_run(task_hash):
    return await agent.run(task=task_hash)

# 4. Reduce context size
context = BufferedChatCompletionContext(buffer_size=5)
```

## Debugging Techniques

### 1. Enable Detailed Logging

```python
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log specific components
logging.getLogger("autogen_core").setLevel(logging.DEBUG)
logging.getLogger("autogen_agentchat").setLevel(logging.DEBUG)
logging.getLogger("autogen_ext").setLevel(logging.DEBUG)
```

### 2. Trace Message Flow

```python
class DebugAgent(AssistantAgent):
    async def on_message(self, message, ctx):
        print(f"[{self.name}] Received: {message}")
        response = await super().on_message(message, ctx)
        print(f"[{self.name}] Sending: {response}")
        return response
```

### 3. Inspect Model Calls

```python
class DebugModelClient:
    def __init__(self, base_client):
        self._client = base_client
    
    async def create(self, messages, **kwargs):
        print("Model Input:", messages)
        response = await self._client.create(messages, **kwargs)
        print("Model Output:", response)
        return response
    
    # Proxy other methods...

# Use debug wrapper
debug_client = DebugModelClient(model_client)
agent = AssistantAgent("agent", model_client=debug_client)
```

## Environment-Specific Issues

### Google Colab

```python
# Install with system pip
!pip install autogen-agentchat autogen-ext[openai]

# Handle async in Colab
import nest_asyncio
nest_asyncio.apply()

# Now you can use await in cells
result = await agent.run(task="Hello")
```

### AWS Lambda

```python
# Lambda handler for AutoGen
import asyncio

def lambda_handler(event, context):
    async def process():
        agent = create_agent()  # Your agent setup
        result = await agent.run(task=event["task"])
        return {"response": result.messages[-1].content}
    
    return asyncio.run(process())
```

### Docker Containers

```dockerfile
FROM python:3.11-slim

# Install AutoGen
RUN pip install autogen-agentchat autogen-ext[docker]

# Install Docker CLI for DockerCodeExecutor
RUN apt-get update && apt-get install -y docker.io

# Your app
COPY . /app
WORKDIR /app

CMD ["python", "main.py"]
```

## Getting Help

### 1. Error Messages

Always include:
- Full error traceback
- AutoGen versions
- Minimal reproducible example
- Environment details (OS, Python version)

### 2. Community Resources

- GitHub Issues: Report bugs
- Discord: Real-time help
- Documentation: Check latest docs
- Examples: Review sample code

### 3. Debug Template

```python
# Minimal reproducible example template
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def debug_issue():
    # Setup
    print(f"Python: {sys.version}")
    print(f"AutoGen: {autogen_agentchat.__version__}")
    
    try:
        # Your problematic code here
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        agent = AssistantAgent("test", model_client)
        result = await agent.run(task="Test")
        print("Success:", result)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(debug_issue())
```

Remember: Most issues have simple solutions. Check the basics first (imports, async/await, API keys) before diving deep!