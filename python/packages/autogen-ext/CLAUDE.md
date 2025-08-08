# CLAUDE.md - AutoGen Extensions Package

*I'll help you integrate the right extensions for your project - from choosing model providers to implementing custom tools and executors.*

This guide helps developers understand how to use AutoGen Extensions for model clients, tools, and other capabilities.

## Overview

The Extensions package (`autogen-ext`) provides implementations for model clients, code executors, tools, and other capabilities that extend the core AutoGen framework. Install specific extensions as needed.

## Installation

```bash
# Install with specific providers
pip install "autogen-ext[openai]"              # OpenAI models
pip install "autogen-ext[anthropic]"           # Anthropic Claude
pip install "autogen-ext[azure]"               # Azure OpenAI
pip install "autogen-ext[gemini]"              # Google Gemini
pip install "autogen-ext[ollama]"              # Local Ollama
pip install "autogen-ext[web-surfer]"          # Web browsing
pip install "autogen-ext[docker]"              # Docker code execution
```

## Model Clients (`autogen_ext.models`)

### OpenAI
```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key="your-key",  # Or use OPENAI_API_KEY env var
    temperature=0.7,
    max_tokens=2000
)
```

### Anthropic Claude
```python
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

client = AnthropicChatCompletionClient(
    model="claude-3-5-sonnet-20241022",
    api_key="your-key",  # Or use ANTHROPIC_API_KEY env var
)
```

### Azure OpenAI
```python
from autogen_ext.models.azure import AzureOpenAIChatCompletionClient

client = AzureOpenAIChatCompletionClient(
    endpoint="https://your-resource.openai.azure.com",
    deployment_name="your-deployment",
    api_version="2024-02-01",
    api_key="your-key"  # Or use DefaultAzureCredential
)
```

### Google Gemini
```python
from autogen_ext.models.gemini import GeminiChatCompletionClient

client = GeminiChatCompletionClient(
    model="gemini-1.5-pro",
    api_key="your-key"  # Or use GOOGLE_API_KEY env var
)
```

### Local Ollama
```python
from autogen_ext.models.ollama import OllamaChatCompletionClient

client = OllamaChatCompletionClient(
    model="llama3.2",
    base_url="http://localhost:11434"  # Default Ollama URL
)
```

### Model Caching
```python
from autogen_ext.models.cache import ChatCompletionCache

# Wrap any client with caching
cached_client = ChatCompletionCache(
    model_client=base_client,
    cache=DiskCacheStore("./cache")  # Or RedisCache, InMemoryCache
)
```

### Testing with Replay
```python
from autogen_ext.models.replay import ReplayChatCompletionClient

# Deterministic responses for testing
test_client = ReplayChatCompletionClient(
    responses=["First response", "Second response"]
)
```

## Code Executors (`autogen_ext.code_executors`)

### Local Command Line
```python
from autogen_ext.code_executors import LocalCommandLineCodeExecutor

executor = LocalCommandLineCodeExecutor(
    work_dir="./workspace",
    # Virtual environment for Python
    virtual_env_context=VirtualEnvContext(
        work_dir="./venv",
        pip_requirements=["numpy", "pandas"]
    )
)
```

### Docker Container
```python
from autogen_ext.code_executors import DockerCommandLineCodeExecutor

executor = DockerCommandLineCodeExecutor(
    image="python:3.11",
    work_dir="/workspace",
    timeout=60,
    mount_dir="./local_workspace"  # Mount local directory
)
```

### Azure Container Apps
```python
from autogen_ext.code_executors.azure import AzureContainerCommandLineCodeExecutor

executor = AzureContainerCommandLineCodeExecutor(
    subscription_id="your-sub-id",
    resource_group="your-rg",
    container_app_env="your-env"
)
```

## Tools (`autogen_ext.tools`)

### MCP (Model Context Protocol) Tools
```python
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

# Connect to MCP server
workbench = McpWorkbench()
await workbench.add_client(
    name="filesystem",
    params=StdioServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )
)

# Get tools from MCP server
tools = await workbench.get_tools()
```

### Tool Adapters
```python
from autogen_ext.tools import LangChainToolAdapter
from langchain_community.tools import WikipediaQueryRun

# Use LangChain tools in AutoGen
langchain_tool = WikipediaQueryRun()
autogen_tool = LangChainToolAdapter(langchain_tool)
```

## Specialized Agents (`autogen_ext.agents`)

### Web Surfer Agent
```python
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

surfer = MultimodalWebSurfer(
    name="web_surfer",
    model_client=model_client,
    headless=False,  # Show browser window
    animate_actions=True,  # Visualize actions
    viewport_size=(1280, 720)
)

# Use in teams
result = await surfer.run(task="Search for AutoGen documentation")
```

### Magentic One
```python
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent

coder = MagenticOneCoderAgent(
    name="coder",
    model_client=model_client,
    instruction="You are an expert Python developer"
)
```

## Memory Systems (`autogen_ext.memory`)

### List Memory
```python
from autogen_ext.memory import ListMemory

memory = ListMemory(name="conversation_history")
await memory.add(MemoryContent(
    content="User prefers concise responses",
    mime_type="text/plain"
))
```

### Chroma DB Vector Memory
```python
from autogen_ext.memory import ChromaMemory

memory = ChromaMemory(
    name="knowledge_base",
    collection_name="documents",
    embedding_model="text-embedding-3-small"
)
```

## Authentication (`autogen_ext.auth`)

### Azure Active Directory
```python
from autogen_ext.auth import AzureAuthProvider

auth = AzureAuthProvider(
    tenant_id="your-tenant",
    client_id="your-client"
)
```

## Cache Stores (`autogen_ext.cache_store`)

### Disk Cache
```python
from autogen_ext.cache_store import DiskCacheStore

cache = DiskCacheStore(cache_dir="./cache")
```

### Redis Cache
```python
from autogen_ext.cache_store import RedisCacheStore

cache = RedisCacheStore(
    host="localhost",
    port=6379,
    password="optional"
)
```

## UI Components (`autogen_ext.ui`)

### Gradio Chat Interface
```python
from autogen_ext.ui import AGSChatInterface

interface = AGSChatInterface(
    team=your_team,
    share=True  # Get public URL
)
interface.launch()
```

## Creating Your Own Extensions

AutoGen 0.4 makes it easy to create and publish your own extensions to the ecosystem.

### Best Practices

#### 1. Naming Convention
Prefix your package name with `autogen-` for better discoverability:
- `autogen-mycompany-models`
- `autogen-custom-tools`
- `autogen-specialized-agents`

#### 2. Implement Common Interfaces
Always implement the standard interfaces from `autogen_core`:

```python
# For model clients
from autogen_core.models import ChatCompletionClient
from autogen_core import Component, ComponentModel

class MyModelClient(ChatCompletionClient, Component[MyModelConfig]):
    component_type = "model_client"
    
    async def create(self, messages, **kwargs):
        # Your implementation
        pass
    
    def create_stream(self, messages, **kwargs):
        # Your streaming implementation
        pass
```

```python
# For tools
from autogen_core.tools import BaseTool
from pydantic import BaseModel

class MyToolArgs(BaseModel):
    param1: str
    param2: int

class MyCustomTool(BaseTool[MyToolArgs, str]):
    def __init__(self):
        super().__init__(
            args_type=MyToolArgs,
            return_type=str,
            name="my_tool",
            description="Does something useful"
        )
    
    async def run(self, args: MyToolArgs) -> str:
        # Your implementation
        return f"Result for {args.param1}"
```

#### 3. Version Dependencies
Specify AutoGen version constraints in `pyproject.toml`:

```toml
[project]
name = "autogen-my-extension"
version = "0.1.0"
dependencies = [
    "autogen-core>=0.4,<0.5",
    "autogen-agentchat>=0.4,<0.5",  # If needed
]

[project.optional-dependencies]
test = ["pytest", "pytest-asyncio"]
```

#### 4. Type Hints
Use comprehensive type hints for better developer experience:

```python
from typing import List, Dict, Optional, AsyncIterator
from autogen_core.models import LLMMessage, CreateResult

async def process_messages(
    messages: List[LLMMessage],
    config: Optional[Dict[str, Any]] = None
) -> CreateResult:
    ...
```

#### 5. Configuration Support
Make your extension configurable via ComponentModel:

```python
from pydantic import BaseModel, Field
from autogen_core import Component, ComponentModel

class MyExtensionConfig(BaseModel):
    api_key: str = Field(description="API key for service")
    timeout: int = Field(default=30, description="Request timeout")
    retry_count: int = Field(default=3, ge=0)

class MyExtension(Component[MyExtensionConfig]):
    component_type = "my_extension"
    component_config_schema = MyExtensionConfig
    
    def __init__(self, api_key: str, timeout: int = 30, retry_count: int = 3):
        self._config = MyExtensionConfig(
            api_key=api_key,
            timeout=timeout,
            retry_count=retry_count
        )
```

### Example Extension Structure

```
autogen-my-extension/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── autogen_my_extension/
│       ├── __init__.py
│       ├── models.py      # Model client implementations
│       ├── tools.py       # Tool implementations
│       ├── agents.py      # Custom agents
│       └── config.py      # Configuration models
├── tests/
│   ├── test_models.py
│   └── test_tools.py
└── examples/
    ├── basic_usage.py
    └── advanced_example.py
```

### Publishing Your Extension

#### 1. Package Setup
Create a proper `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "autogen-my-extension"
description = "My custom AutoGen extension"
readme = "README.md"
license = "MIT"
authors = [{name = "Your Name", email = "you@example.com"}]
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
```

#### 2. Discovery
Add GitHub topics for discoverability:
- `autogen` - General AutoGen ecosystem
- `autogen-extension` - For extensions
- `autogen-sample` - For example projects

#### 3. Documentation
Include clear examples in your README:

```markdown
# AutoGen My Extension

Custom model client and tools for AutoGen.

## Installation
```bash
pip install autogen-my-extension
```

## Quick Start
```python
from autogen_my_extension import MyModelClient
from autogen_agentchat.agents import AssistantAgent

client = MyModelClient(api_key="...")
agent = AssistantAgent("assistant", model_client=client)
```
```

### Integration Testing

Test your extension with AutoGen:

```python
import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_my_extension import MyModelClient

@pytest.mark.asyncio
async def test_with_assistant_agent():
    client = MyModelClient(api_key="test")
    agent = AssistantAgent("test", model_client=client)
    
    result = await agent.run(task="Hello")
    assert result.messages[-1].content == "Expected response"
```

### Common Extension Types

1. **Model Clients**: New LLM providers
2. **Tools**: API integrations, data processing
3. **Code Executors**: New execution environments
4. **Memory Systems**: Vector stores, databases
5. **Agents**: Specialized agent behaviors
6. **UI Components**: Custom interfaces

Remember: Focus on implementing standard interfaces to ensure compatibility with the broader AutoGen ecosystem.