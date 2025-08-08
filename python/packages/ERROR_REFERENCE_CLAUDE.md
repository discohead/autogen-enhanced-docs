# CLAUDE.md - AutoGen Error Reference

*As your AutoGen guide, I'll help you understand and resolve errors quickly. This reference catalogs common errors with their causes and solutions.*

This comprehensive error reference helps you diagnose and fix AutoGen errors efficiently.

## Error Categories

### Import and Installation Errors

#### ImportError: No module named 'autogen_agentchat'

**Cause**: Package not installed or wrong package name

**Solutions**:
```bash
# Correct installation
pip install autogen-agentchat

# Common mistakes:
# ❌ pip install autogen (old version)
# ❌ pip install pyautogen (old version)
# ✅ pip install autogen-agentchat
```

#### ImportError: cannot import name 'OpenAIChatCompletionClient'

**Cause**: Missing extension package

**Solutions**:
```bash
# Install with specific extension
pip install "autogen-ext[openai]"

# Or multiple extensions
pip install "autogen-ext[openai,anthropic,docker]"
```

#### ModuleNotFoundError: No module named 'autogen_core'

**Cause**: Core package not installed (usually installed as dependency)

**Solutions**:
```bash
# Explicitly install core
pip install autogen-core

# Or reinstall everything
pip install --upgrade autogen-agentchat autogen-ext[openai]
```

### Async/Await Errors

#### RuntimeWarning: coroutine '...' was never awaited

**Cause**: Missing `await` keyword

**Example**:
```python
# ❌ Wrong
result = agent.run(task="Hello")

# ✅ Correct
result = await agent.run(task="Hello")
```

#### SyntaxError: 'await' outside async function

**Cause**: Using await in non-async context

**Solutions**:
```python
# ❌ Wrong
def main():
    result = await agent.run(task="Hello")

# ✅ Correct
async def main():
    result = await agent.run(task="Hello")

# Run with asyncio
import asyncio
asyncio.run(main())
```

#### RuntimeError: asyncio.run() cannot be called from a running event loop

**Cause**: Nested event loops (common in Jupyter)

**Solutions**:
```python
# For Jupyter/IPython
import nest_asyncio
nest_asyncio.apply()

# Or use await directly (Jupyter supports top-level await)
result = await agent.run(task="Hello")
```

### Model Client Errors

#### openai.AuthenticationError: No API key provided

**Cause**: Missing or invalid API key

**Solutions**:
```python
# Set environment variable
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# Or pass directly (not recommended for production)
client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key="sk-..."
)

# Using .env file
from dotenv import load_dotenv
load_dotenv()
```

#### openai.NotFoundError: The model 'gpt-4' does not exist

**Cause**: No access to requested model

**Solutions**:
```python
# Use available model
client = OpenAIChatCompletionClient(
    model="gpt-3.5-turbo"  # If no GPT-4 access
)

# Check available models for your account
# Try different models based on your access level
```

#### openai.RateLimitError: Rate limit reached

**Cause**: Too many requests

**Solutions**:
```python
# Add retry logic
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3)
)
async def call_model():
    return await agent.run(task="...")

# Or use built-in retries
client = OpenAIChatCompletionClient(
    model="gpt-4o",
    max_retries=3,
    retry_delay=1.0
)
```

#### anthropic.BadRequestError: max_tokens required

**Cause**: Missing required parameter for Anthropic

**Solutions**:
```python
# Anthropic requires max_tokens
client = AnthropicChatCompletionClient(
    model="claude-3-5-sonnet-20241022",
    api_key="...",
    max_tokens=4000  # Required!
)
```

### Agent Errors

#### ValueError: Agent name cannot be empty

**Cause**: Missing agent name

**Solutions**:
```python
# ❌ Wrong
agent = AssistantAgent(model_client=client)

# ✅ Correct
agent = AssistantAgent(
    name="assistant",  # Required!
    model_client=client
)
```

#### RuntimeError: No termination condition set

**Cause**: Team without termination condition

**Solutions**:
```python
# ❌ Wrong
team = RoundRobinGroupChat([agent1, agent2])

# ✅ Correct
from autogen_agentchat.conditions import MaxMessageTermination

team = RoundRobinGroupChat(
    participants=[agent1, agent2],
    termination_condition=MaxMessageTermination(10)
)
```

#### ValueError: Unknown tool

**Cause**: Tool not properly registered

**Solutions**:
```python
# Ensure tool is created and added
from autogen_core.tools import FunctionTool

def my_tool(param: str) -> str:
    """Tool description required!"""
    return f"Result: {param}"

tool = FunctionTool(my_tool, name="my_tool")

agent = AssistantAgent(
    "agent",
    model_client=client,
    tools=[tool]  # Add tool here
)
```

### Memory and Context Errors

#### openai.BadRequestError: maximum context length is 4096 tokens

**Cause**: Context window exceeded

**Solutions**:
```python
# Use model with larger context
client = OpenAIChatCompletionClient(
    model="gpt-4-turbo"  # 128k context
)

# Or limit context
from autogen_core.model_context import BufferedChatCompletionContext

context = BufferedChatCompletionContext(
    buffer_size=10  # Keep only last 10 messages
)

agent = AssistantAgent(
    "agent",
    model_client=client,
    model_context=context
)
```

#### MemoryError: Unable to allocate array

**Cause**: Out of memory

**Solutions**:
```python
# Reduce batch sizes
# Process data in chunks
# Use streaming where possible

# For teams, limit message history
team = RoundRobinGroupChat(
    participants=[...],
    max_message_history=50  # Limit stored messages
)
```

### Tool and Function Errors

#### pydantic.ValidationError: Invalid tool schema

**Cause**: Improper type hints or missing docstring

**Solutions**:
```python
# ❌ Wrong
def my_tool(param):  # No type hints
    return "result"

# ✅ Correct
def my_tool(param: str) -> str:
    """Tool description.
    
    Args:
        param: Parameter description
    """
    return f"result: {param}"
```

#### TypeError: Tool function must be callable

**Cause**: Passing non-function to FunctionTool

**Solutions**:
```python
# ❌ Wrong
tool = FunctionTool("not_a_function")

# ✅ Correct
def actual_function():
    return "result"

tool = FunctionTool(actual_function)
```

### Code Executor Errors

#### docker.errors.DockerException: Error while fetching server API version

**Cause**: Docker not running or not installed

**Solutions**:
```bash
# Start Docker
# macOS: open -a Docker
# Linux: sudo systemctl start docker
# Windows: Start Docker Desktop

# Or use local executor instead
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
executor = LocalCommandLineCodeExecutor(work_dir="./workspace")
```

#### TimeoutError: Code execution timed out

**Cause**: Code took too long to execute

**Solutions**:
```python
# Increase timeout
executor = DockerCodeExecutor(
    image="python:3.11",
    timeout=120  # 2 minutes instead of default
)

# Or handle in code
try:
    result = await executor.execute_code(code, "python")
except TimeoutError:
    print("Execution timed out")
```

#### PermissionError: [Errno 13] Permission denied

**Cause**: Insufficient permissions for file operations

**Solutions**:
```python
# Check work directory permissions
import os
os.chmod("./workspace", 0o755)

# Or use different directory
executor = LocalCommandLineCodeExecutor(
    work_dir="/tmp/autogen_workspace"  # Usually writable
)
```

### Team and Communication Errors

#### ValueError: No participants in team

**Cause**: Empty participant list

**Solutions**:
```python
# ❌ Wrong
team = RoundRobinGroupChat(participants=[])

# ✅ Correct
team = RoundRobinGroupChat(
    participants=[agent1, agent2]  # At least one agent
)
```

#### RuntimeError: Circular dependency detected

**Cause**: Agents referring to each other incorrectly

**Solutions**:
```python
# Avoid circular handoffs
# ❌ Wrong
agent1.handoffs = [Handoff(target="agent2")]
agent2.handoffs = [Handoff(target="agent1")]

# ✅ Better
agent1.handoffs = [Handoff(target="agent2", condition="specific_case")]
agent2.handoffs = [Handoff(target="agent3")]
```

### Configuration Errors

#### yaml.YAMLError: could not determine a constructor

**Cause**: Invalid YAML syntax

**Solutions**:
```yaml
# Check YAML syntax
# Use proper indentation (spaces, not tabs)
# Quote special characters
# Validate with online YAML validator
```

#### KeyError: 'OPENAI_API_KEY'

**Cause**: Environment variable not set

**Solutions**:
```python
# Provide default or handle missing
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Or use default
api_key = os.getenv("OPENAI_API_KEY", "default-key")
```

### Network and Connection Errors

#### aiohttp.ClientConnectorError: Cannot connect to host

**Cause**: Network connectivity issues

**Solutions**:
```python
# Check internet connection
# Verify firewall settings
# Try with proxy if needed

import aiohttp
connector = aiohttp.TCPConnector(
    ssl=False,  # For testing only
    limit=10
)
```

#### ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]

**Cause**: SSL certificate verification failed

**Solutions**:
```python
# For testing only (not for production!)
import ssl
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Better: Update certificates
# macOS: brew install ca-certificates
# Update pip: pip install --upgrade certifi
```

## Error Patterns and Debugging

### Stack Trace Analysis

```python
# Example error analysis
try:
    result = await agent.run(task="Complex task")
except Exception as e:
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
    
    # Get full traceback
    import traceback
    traceback.print_exc()
    
    # Get specific error details
    if hasattr(e, 'response'):
        print(f"Response: {e.response}")
    if hasattr(e, 'status_code'):
        print(f"Status Code: {e.status_code}")
```

### Common Error Chains

```
ImportError → ModuleNotFoundError → Package not installed
AuthenticationError → 401 Error → Invalid or missing API key
RateLimitError → 429 Error → Too many requests
ContextLengthError → 400 Error → Message too long
TimeoutError → asyncio.TimeoutError → Operation took too long
```

### Debug Mode

```python
# Enable debug logging for detailed errors
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Component-specific debugging
logging.getLogger("autogen_core").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)
```

## Quick Error Lookup

| Error Keyword | Likely Cause | Quick Fix |
|--------------|--------------|-----------|
| "import" | Missing package | `pip install autogen-agentchat` |
| "await" | Async issue | Add `await` or use `asyncio.run()` |
| "API key" | Auth problem | Set environment variable |
| "not found" | Wrong model/resource | Check availability |
| "rate limit" | Too many requests | Add delays or retry |
| "context length" | Too much data | Use larger model or reduce input |
| "timeout" | Slow operation | Increase timeout |
| "docker" | Container issue | Start Docker service |
| "permission" | File access | Check directory permissions |
| "YAML" | Config syntax | Validate YAML format |

## Preventive Measures

### Input Validation

```python
def validate_agent_config(name: str, model_client, system_message: str):
    if not name:
        raise ValueError("Agent name required")
    if not model_client:
        raise ValueError("Model client required")
    if len(system_message) > 1000:
        raise ValueError("System message too long")
```

### Error Boundaries

```python
async def safe_agent_run(agent, task):
    try:
        return await agent.run(task=task)
    except RateLimitError:
        await asyncio.sleep(60)  # Wait a minute
        return await agent.run(task=task)
    except ContextLengthError:
        # Retry with shorter task
        return await agent.run(task=task[:1000])
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None
```

### Health Checks

```python
async def check_system_health():
    checks = {
        "model_client": False,
        "docker": False,
        "network": False
    }
    
    # Check model client
    try:
        await model_client.create([{"role": "user", "content": "test"}])
        checks["model_client"] = True
    except:
        pass
    
    # Check Docker
    try:
        import docker
        client = docker.from_env()
        client.ping()
        checks["docker"] = True
    except:
        pass
    
    # Check network
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.openai.com") as resp:
                checks["network"] = resp.status == 200
    except:
        pass
    
    return checks
```

Remember: Most errors have straightforward solutions. Check the error message carefully, it usually points to the exact problem!