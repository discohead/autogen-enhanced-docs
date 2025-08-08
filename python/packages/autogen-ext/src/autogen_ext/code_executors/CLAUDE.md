# CLAUDE.md - AutoGen Code Executors Guide

*As your AutoGen guide, I'll help you understand code executors - the secure sandboxes that let your agents safely run code. Choose the right executor for your security and performance needs.*

This guide covers all code execution environments available in AutoGen Extensions, helping you enable safe code execution for your agents.

## Overview

Code executors provide secure, isolated environments for agents to run code. They're essential for CodeExecutorAgent and any agent that needs to execute dynamic code. AutoGen offers multiple executors with different isolation levels and capabilities.

## Available Code Executors

### 1. LocalCommandLineCodeExecutor

Executes code directly on the local machine (least isolated).

```python
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

# Basic local executor
executor = LocalCommandLineCodeExecutor(
    work_dir="./workspace",  # Working directory for code files
    executors=["python", "bash", "javascript"],  # Allowed languages
    timeout=30  # Timeout in seconds
)

# Use with CodeExecutorAgent
from autogen_agentchat.agents import CodeExecutorAgent
code_agent = CodeExecutorAgent(
    "python_executor",
    code_executor=executor
)
```

**Advanced Configuration**:

```python
# With virtual environment
executor = LocalCommandLineCodeExecutor(
    work_dir="./workspace",
    executors={
        "python": {
            "executable": "/path/to/venv/bin/python",
            "file_extension": ".py"
        },
        "node": {
            "executable": "/usr/local/bin/node",
            "file_extension": ".js"
        }
    },
    # Execution options
    timeout=60,
    max_memory_mb=512,
    env_vars={"PYTHONPATH": "/custom/path"}
)
```

**Security Considerations**:
- ⚠️ **No isolation** - code runs with your user permissions
- ⚠️ Can access local file system
- ⚠️ Can make network requests
- ✅ Fast execution
- ✅ No additional setup required

**When to Use**:
- Development and testing
- Trusted code only
- When you need local system access
- Quick prototyping

### 2. DockerCodeExecutor

Executes code in Docker containers (recommended for production).

```python
from autogen_ext.code_executors.docker import DockerCodeExecutor

# Basic Docker executor
executor = DockerCodeExecutor(
    image="python:3.11-slim",  # Docker image
    work_dir="/workspace",     # Container work directory
    timeout=60,
    auto_remove=True  # Remove container after execution
)

# Multi-language support
executor = DockerCodeExecutor(
    image="custom-multi-lang:latest",  # Image with multiple languages
    executors=["python", "javascript", "ruby"],
    volumes={
        "/host/data": {
            "bind": "/container/data",
            "mode": "ro"  # Read-only mount
        }
    },
    environment={"API_KEY": "safe-key"}
)
```

**Custom Docker Images**:

```dockerfile
# Dockerfile for custom executor image
FROM python:3.11-slim

# Install additional languages
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    ruby \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install numpy pandas matplotlib

# Create work directory
WORKDIR /workspace
```

```python
# Use custom image
executor = DockerCodeExecutor(
    image="my-executor:latest",
    build_path="./docker",  # Build from Dockerfile
    resource_limits={
        "memory": "512m",
        "cpus": "1.0",
        "pids": 100
    }
)
```

**Security Benefits**:
- ✅ **Full isolation** via containers
- ✅ Resource limits (CPU, memory)
- ✅ Network isolation options
- ✅ Temporary file systems
- ⚠️ Requires Docker installation

**When to Use**:
- Production environments
- Untrusted code execution
- Multi-user systems
- When you need resource limits

### 3. JupyterCodeExecutor

Executes code in Jupyter kernels with persistent state.

```python
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor

# Connect to existing Jupyter server
executor = JupyterCodeExecutor(
    server_url="http://localhost:8888",
    token="your-jupyter-token",
    kernel_name="python3"
)

# Local Jupyter kernel
executor = JupyterCodeExecutor(
    kernel_name="python3",
    timeout=60,
    startup_timeout=30
)

# Use for stateful execution
code_agent = CodeExecutorAgent(
    "notebook_executor",
    code_executor=executor
)

# Code persists between executions
await code_agent.run(task="import pandas as pd")
await code_agent.run(task="df = pd.DataFrame({'a': [1,2,3]})")
await code_agent.run(task="print(df.head())")  # df still exists
```

**Advanced Features**:

```python
# With custom kernel
executor = JupyterCodeExecutor(
    kernel_name="custom_env",
    kernel_spec={
        "display_name": "Custom Environment",
        "language": "python",
        "argv": [
            "/path/to/custom/python",
            "-m", "ipykernel_launcher",
            "-f", "{connection_file}"
        ]
    }
)

# Multiple language support
executor = JupyterCodeExecutor(
    kernel_manager_class="multilang.MultiKernelManager",
    supported_kernels=["python3", "ir", "julia-1.8"]
)
```

**Key Features**:
- ✅ **Persistent state** between executions
- ✅ Rich output support (plots, HTML)
- ✅ Interactive development
- ✅ Multiple kernel support
- ⚠️ Less isolation than Docker

**When to Use**:
- Data science workflows
- Interactive development
- When you need persistent state
- Educational environments

### 4. DockerJupyterCodeExecutor

Combines Docker isolation with Jupyter functionality.

```python
from autogen_ext.code_executors.docker_jupyter import DockerJupyterCodeExecutor

# Jupyter server in Docker
executor = DockerJupyterCodeExecutor(
    image="jupyter/scipy-notebook:latest",
    port_range=(8888, 8900),  # Port range for containers
    timeout=60,
    kernel_name="python3",
    volumes={"/data": {"bind": "/home/jovyan/data", "mode": "rw"}}
)

# Custom Jupyter Docker image
executor = DockerJupyterCodeExecutor(
    image="custom-jupyter:latest",
    resource_limits={
        "memory": "2g",
        "cpus": "2.0"
    },
    jupyter_args=["--NotebookApp.token=''"],  # No token
    startup_timeout=120  # Longer startup for heavy images
)
```

**Best of Both Worlds**:
- ✅ Docker isolation
- ✅ Jupyter state persistence
- ✅ Resource limits
- ✅ Rich outputs
- ⚠️ Higher resource usage

**When to Use**:
- Production data science
- Multi-tenant environments
- When you need both isolation and state
- Complex analytical workflows

### 5. AzureContainerCodeExecutor

Executes code in Azure Container Instances.

```python
from autogen_ext.code_executors.azure import AzureContainerCodeExecutor

# Azure Container executor
executor = AzureContainerCodeExecutor(
    subscription_id="your-subscription-id",
    resource_group="your-resource-group",
    container_group_name="autogen-executor",
    image="python:3.11-slim",
    location="eastus",
    cpu_cores=1.0,
    memory_gb=1.5,
    timeout=300
)

# With Azure credentials
from azure.identity import DefaultAzureCredential

executor = AzureContainerCodeExecutor(
    credential=DefaultAzureCredential(),
    subscription_id="...",
    resource_group="...",
    managed_identity_resource_id="/subscriptions/.../resourceGroups/.../providers/Microsoft.ManagedIdentity/...",
    azure_file_share={
        "account_name": "storage_account",
        "share_name": "code_share",
        "mount_path": "/data"
    }
)
```

**Cloud Benefits**:
- ✅ **Serverless execution**
- ✅ Auto-scaling
- ✅ No local resources needed
- ✅ Enterprise security
- ⚠️ Network latency
- ⚠️ Cost considerations

**When to Use**:
- Cloud-native applications
- Serverless architectures
- Enterprise environments
- When local resources are limited

## Code Execution Patterns

### Basic Execution

```python
# Direct code execution
result = await executor.execute_code(
    code="print('Hello, World!')",
    language="python"
)
print(result.output)  # "Hello, World!\n"
print(result.exit_code)  # 0
```

### Multi-Language Execution

```python
# Python code
py_result = await executor.execute_code(
    code="""
import json
data = {"result": 42}
print(json.dumps(data))
""",
    language="python"
)

# JavaScript code
js_result = await executor.execute_code(
    code="""
const data = {result: 42};
console.log(JSON.stringify(data));
""",
    language="javascript"
)
```

### File-Based Execution

```python
# Save and execute files
code_with_files = """
# main.py
import helper

result = helper.calculate(10, 20)
print(f"Result: {result}")
"""

helper_code = """
# helper.py
def calculate(a, b):
    return a + b
"""

# Execute with multiple files
result = await executor.execute_code(
    files={
        "main.py": code_with_files,
        "helper.py": helper_code
    },
    entry_point="main.py",
    language="python"
)
```

### Handling Outputs

```python
# Capture different output streams
result = await executor.execute_code(
    code="""
import sys
print("Standard output")
print("Error output", file=sys.stderr)
exit(1)
""",
    language="python"
)

print(f"STDOUT: {result.output}")
print(f"STDERR: {result.error}")
print(f"Exit Code: {result.exit_code}")
```

## Security Best Practices

### 1. Choose the Right Isolation Level

```python
# Development - Local executor is fine
if environment == "development":
    executor = LocalCommandLineCodeExecutor(work_dir="./dev")

# Production - Use Docker or Azure
elif environment == "production":
    executor = DockerCodeExecutor(
        image="secure-executor:latest",
        network_mode="none",  # No network access
        read_only=True,       # Read-only file system
        resource_limits={"memory": "256m", "cpus": "0.5"}
    )
```

### 2. Resource Limits

```python
# Always set resource limits
executor = DockerCodeExecutor(
    image="python:3.11-slim",
    resource_limits={
        "memory": "512m",      # Memory limit
        "memory_swap": "512m", # Prevent swap usage
        "cpus": "1.0",        # CPU limit
        "pids": 50            # Process limit
    },
    timeout=30  # Execution timeout
)
```

### 3. Input Validation

```python
# Validate code before execution
def validate_code(code: str, language: str) -> bool:
    # Check for dangerous patterns
    dangerous_patterns = [
        r"import\s+os",
        r"import\s+subprocess",
        r"__import__",
        r"eval\s*\(",
        r"exec\s*\("
    ]
    
    if language == "python":
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return False
    
    return True

# Use validation
if validate_code(user_code, "python"):
    result = await executor.execute_code(user_code, "python")
else:
    raise ValueError("Code contains prohibited patterns")
```

### 4. Network Isolation

```python
# Disable network access
executor = DockerCodeExecutor(
    image="python:3.11-slim",
    network_mode="none",  # Complete network isolation
    # Or restrict to specific networks
    network_mode="isolated_network",
    dns=[]  # No DNS resolution
)
```

## Performance Optimization

### 1. Container Reuse

```python
# Reuse containers for better performance
executor = DockerCodeExecutor(
    image="python:3.11-slim",
    reuse_container=True,  # Keep container running
    container_name="persistent-executor",
    idle_timeout=300  # Remove after 5 minutes idle
)
```

### 2. Pre-warmed Executors

```python
# Pre-warm executors
async def create_executor_pool(count: int = 5):
    executors = []
    for i in range(count):
        executor = DockerCodeExecutor(
            image="python:3.11-slim",
            container_name=f"executor-{i}"
        )
        await executor.start()  # Pre-start container
        executors.append(executor)
    return executors

# Use from pool
executor_pool = await create_executor_pool()
executor = executor_pool.pop()
result = await executor.execute_code(code, "python")
executor_pool.append(executor)  # Return to pool
```

### 3. Caching Results

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
async def cached_execute(code_hash: str, executor):
    # Execute only if not in cache
    return await executor.execute_code(code, "python")

# Use caching
code_hash = hashlib.md5(code.encode()).hexdigest()
result = await cached_execute(code_hash, executor)
```

## Error Handling

### Common Errors and Solutions

```python
# Timeout handling
try:
    result = await executor.execute_code(
        long_running_code,
        language="python",
        timeout=10
    )
except TimeoutError:
    print("Code execution timed out")
    # Try with longer timeout or optimize code

# Resource limit errors
try:
    result = await executor.execute_code(
        memory_intensive_code,
        language="python"
    )
except ResourceExhaustedError as e:
    print(f"Resource limit hit: {e}")
    # Increase limits or optimize code

# Language not supported
try:
    result = await executor.execute_code(
        code="println('Hello')",
        language="kotlin"
    )
except LanguageNotSupportedError:
    print("Kotlin not supported in this executor")
```

### Debugging Execution

```python
# Enable debug logging
import logging
logging.getLogger("autogen_ext.code_executors").setLevel(logging.DEBUG)

# Debug wrapper
class DebugExecutor:
    def __init__(self, executor):
        self._executor = executor
    
    async def execute_code(self, code, language, **kwargs):
        print(f"Executing {language} code:")
        print("-" * 40)
        print(code)
        print("-" * 40)
        
        result = await self._executor.execute_code(code, language, **kwargs)
        
        print(f"Output: {result.output}")
        print(f"Error: {result.error}")
        print(f"Exit Code: {result.exit_code}")
        
        return result

# Use debug wrapper
debug_executor = DebugExecutor(executor)
```

## Testing Code Executors

```python
import pytest

@pytest.mark.asyncio
async def test_executor_basic():
    executor = LocalCommandLineCodeExecutor(work_dir="./test")
    
    result = await executor.execute_code(
        code="print('test')",
        language="python"
    )
    
    assert result.exit_code == 0
    assert "test" in result.output

@pytest.mark.asyncio 
async def test_executor_timeout():
    executor = DockerCodeExecutor(image="python:3.11-slim", timeout=1)
    
    with pytest.raises(TimeoutError):
        await executor.execute_code(
            code="import time; time.sleep(10)",
            language="python"
        )
```

## Comparison Matrix

| Executor | Isolation | State | Speed | Setup | Best For |
|----------|-----------|-------|-------|--------|----------|
| Local | None | No | Fast | Easy | Development |
| Docker | High | No | Medium | Medium | Production |
| Jupyter | Low | Yes | Fast | Medium | Data Science |
| DockerJupyter | High | Yes | Slow | Complex | Secure DS |
| Azure | High | No | Slow | Complex | Enterprise |

## Related Resources

- Agent integration: `autogen_agentchat.agents.CodeExecutorAgent`
- Security patterns: See security best practices above
- Docker images: Official language images recommended
- Azure setup: Azure Container Instances documentation

Remember: Always choose the most restrictive executor that meets your needs. Start with Docker for production use!