# CLAUDE.md - AutoGen Model Clients

*I'll guide you through choosing and configuring the right model client for your needs, from setup to optimization tips for each provider.*

This guide provides detailed information about model client implementations in AutoGen Extensions.

## Overview

Model clients in AutoGen implement the `ChatCompletionClient` interface from `autogen_core`. Each client provides access to different LLM providers while maintaining a consistent API.

## Core Interface

All model clients implement:

```python
from autogen_core.models import ChatCompletionClient

class ModelClient(ChatCompletionClient):
    async def create(self, messages, **kwargs) -> CreateResult:
        """Single completion"""
        pass
    
    def create_stream(self, messages, **kwargs) -> AsyncIterator[CreateResult | str]:
        """Streaming completion"""
        pass
    
    @property
    def model_info(self) -> ModelInfo:
        """Model capabilities"""
        pass
```

## OpenAI Client

### Basic Usage
```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key="sk-...",  # Or use OPENAI_API_KEY env var
    base_url=None,  # Optional custom endpoint
    temperature=0.7,
    max_tokens=2000,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={"type": "text"},  # or "json_object"
    seed=None,  # For deterministic outputs
    n=1  # Number of completions
)
```

### Structured Output
```python
from pydantic import BaseModel

class ResponseSchema(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

client = OpenAIChatCompletionClient(
    model="gpt-4o",
    response_format=ResponseSchema  # Pydantic model for structured output
)

result = await client.create(messages)
parsed = ResponseSchema.model_validate_json(result.content)
```

### Vision Support
```python
from autogen_core import Image

messages = [
    UserMessage(
        content=[
            "What's in this image?",
            Image.from_url("https://example.com/image.jpg")
        ],
        source="user"
    )
]

result = await client.create(messages)
```

### Function Calling
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

result = await client.create(
    messages,
    tools=tools,
    tool_choice="auto"  # or "none", "required", {"type": "function", "function": {"name": "get_weather"}}
)

if result.finish_reason == "function_calls":
    for call in result.content:
        print(f"Call {call.name} with {call.arguments}")
```

## Anthropic Client

### Basic Usage
```python
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

client = AnthropicChatCompletionClient(
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",  # Or use ANTHROPIC_API_KEY env var
    base_url=None,
    temperature=0.7,
    max_tokens=4096,
    top_p=1.0,
    top_k=None,  # Anthropic-specific
    stop_sequences=None
)
```

### System Messages
```python
# Anthropic handles system messages differently
messages = [
    SystemMessage(content="You are a helpful assistant"),
    UserMessage(content="Hello", source="user")
]

# Client automatically formats for Anthropic API
result = await client.create(messages)
```

### Vision Support
```python
# Anthropic supports images
messages = [
    UserMessage(
        content=[
            "Analyze this chart",
            Image.from_file("chart.png")
        ],
        source="user"
    )
]
```

## Azure OpenAI Client

### Basic Usage
```python
from autogen_ext.models.azure import AzureOpenAIChatCompletionClient

client = AzureOpenAIChatCompletionClient(
    endpoint="https://myresource.openai.azure.com",
    deployment_name="gpt-4o-deployment",
    api_version="2024-02-01",
    api_key="...",  # Or use AZURE_OPENAI_API_KEY
    # Or use DefaultAzureCredential
    azure_ad_token_provider=None,
    model="gpt-4o",  # Model name for info
    # All OpenAI parameters supported
    temperature=0.7
)
```

### With Managed Identity
```python
from azure.identity import DefaultAzureCredential
from azure.identity import get_bearer_token_provider

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, 
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAIChatCompletionClient(
    endpoint="https://myresource.openai.azure.com",
    deployment_name="gpt-4o",
    api_version="2024-02-01",
    azure_ad_token_provider=token_provider
)
```

## Gemini Client

### Basic Usage
```python
from autogen_ext.models.gemini import GeminiChatCompletionClient

client = GeminiChatCompletionClient(
    model="gemini-1.5-pro",
    api_key="...",  # Or use GOOGLE_API_KEY
    temperature=0.7,
    max_tokens=2048,
    top_p=0.95,
    top_k=40,  # Gemini-specific
    stop_sequences=None,
    response_mime_type="text/plain",  # or "application/json"
    response_schema=None  # For structured output
)
```

### Safety Settings
```python
client = GeminiChatCompletionClient(
    model="gemini-1.5-flash",
    safety_settings={
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_LOW_AND_ABOVE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
    }
)
```

## Ollama Client

### Basic Usage
```python
from autogen_ext.models.ollama import OllamaChatCompletionClient

client = OllamaChatCompletionClient(
    model="llama3.2",
    base_url="http://localhost:11434",  # Default Ollama URL
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
    num_ctx=4096,  # Context window
    seed=None
)
```

### Custom Models
```python
# Use custom Ollama models
client = OllamaChatCompletionClient(
    model="my-custom-model:latest",
    base_url="http://ollama-server:11434"
)
```

## Model Caching

### In-Memory Cache
```python
from autogen_ext.models.cache import ChatCompletionCache, InMemoryCacheStore

cache_store = InMemoryCacheStore()
cached_client = ChatCompletionCache(
    model_client=base_client,
    cache=cache_store
)

# First call hits API
result1 = await cached_client.create(messages)

# Identical call uses cache
result2 = await cached_client.create(messages)
```

### Disk Cache
```python
from autogen_ext.models.cache import DiskCacheStore

cache_store = DiskCacheStore(cache_dir="./model_cache")
cached_client = ChatCompletionCache(
    model_client=base_client,
    cache=cache_store
)
```

### Redis Cache
```python
from autogen_ext.models.cache import RedisCacheStore

cache_store = RedisCacheStore(
    host="localhost",
    port=6379,
    ttl=3600  # 1 hour TTL
)
cached_client = ChatCompletionCache(
    model_client=base_client,
    cache=cache_store
)
```

## Replay Client (Testing)

```python
from autogen_ext.models.replay import ReplayChatCompletionClient

# Fixed responses for testing
test_client = ReplayChatCompletionClient(
    responses=[
        "First response",
        CreateResult(
            content=[FunctionCall(id="1", name="tool", arguments="{}")],
            finish_reason="function_calls"
        ),
        "Tool executed successfully"
    ],
    model_info={
        "model": "test-model",
        "function_calling": True,
        "vision": False,
        "json_output": True
    }
)
```

## Advanced Patterns

### Retry and Rate Limiting
```python
import httpx

# Custom HTTP client with retry
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0),
    limits=httpx.Limits(max_connections=10),
    transport=httpx.AsyncHTTPTransport(retries=3)
)

client = OpenAIChatCompletionClient(
    model="gpt-4o",
    http_client=http_client
)
```

### Model Fallback
```python
class FallbackModelClient(ChatCompletionClient):
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
    
    async def create(self, messages, **kwargs):
        try:
            return await self.primary.create(messages, **kwargs)
        except Exception as e:
            logger.warning(f"Primary failed: {e}, using fallback")
            return await self.fallback.create(messages, **kwargs)
    
    @property
    def model_info(self):
        return self.primary.model_info

# Use GPT-4 with GPT-3.5 fallback
client = FallbackModelClient(
    primary=OpenAIChatCompletionClient(model="gpt-4o"),
    fallback=OpenAIChatCompletionClient(model="gpt-3.5-turbo")
)
```

### Token Counting
```python
class TokenCountingClient(ChatCompletionClient):
    def __init__(self, client):
        self.client = client
        self.total_tokens = 0
    
    async def create(self, messages, **kwargs):
        result = await self.client.create(messages, **kwargs)
        if result.usage:
            self.total_tokens += result.usage.total_tokens
        return result
```

### Streaming with Processing
```python
async def process_stream(client, messages):
    full_content = ""
    
    async for chunk in client.create_stream(messages):
        if isinstance(chunk, str):
            full_content += chunk
            # Process incrementally
            if chunk.endswith("."):
                await process_sentence(full_content)
        else:
            # Final result
            return chunk
```

## Error Handling

```python
from autogen_core.exceptions import ModelClientError
import asyncio

async def robust_completion(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.create(messages)
        except ModelClientError as e:
            if e.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            elif e.status_code >= 500:  # Server error
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
            raise
```

## Performance Tips

1. **Connection Pooling**: Reuse HTTP clients
2. **Caching**: Cache repeated queries
3. **Streaming**: Use for long responses
4. **Batch Requests**: Group when possible
5. **Context Management**: Limit message history
6. **Model Selection**: Balance cost vs performance

## Best Practices

1. **Environment Variables**: Store API keys securely
2. **Error Handling**: Implement retry logic
3. **Logging**: Track API usage
4. **Testing**: Use ReplayClient for tests
5. **Type Safety**: Use Pydantic models
6. **Resource Cleanup**: Close clients when done

```python
# Proper cleanup
async with client:
    result = await client.create(messages)
# Or explicitly
await client.close()
```

Remember: Model clients are the interface between AutoGen and LLMs. Choose the right client and configuration for your use case.