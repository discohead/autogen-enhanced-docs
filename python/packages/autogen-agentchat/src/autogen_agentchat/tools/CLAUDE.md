# CLAUDE.md - AutoGen Tool Integration Guide

*As your AutoGen mentor, I'll help you understand how to create and integrate tools that empower your agents with external capabilities. Tools are the bridge between agents and the real world.*

This guide covers tool patterns in AutoGen AgentChat, helping you add powerful capabilities to your agents.

## Overview

Tools in AutoGen allow agents to interact with external systems, APIs, and perform specialized tasks. The AgentChat package provides high-level tool patterns that work seamlessly with agents.

## Core Tool Types

### 1. AgentTool

Wraps an agent as a tool that can be called by other agents.

```python
from autogen_agentchat.tools import AgentTool
from autogen_agentchat.agents import AssistantAgent

# Create a specialized agent
calculator_agent = AssistantAgent(
    "calculator",
    model_client=model_client,
    system_message="You are a calculator. Solve math problems precisely."
)

# Wrap it as a tool
calculator_tool = AgentTool(
    agent=calculator_agent,
    description="Performs complex calculations",
    name="calculator"
)

# Use in another agent
main_agent = AssistantAgent(
    "main",
    model_client=model_client,
    tools=[calculator_tool]
)
```

### 2. TeamTool

Wraps an entire team as a tool, enabling hierarchical agent systems.

```python
from autogen_agentchat.tools import TeamTool
from autogen_agentchat.teams import RoundRobinGroupChat

# Create a research team
research_team = RoundRobinGroupChat([
    data_analyst,
    researcher,
    summarizer
])

# Wrap team as a tool
research_tool = TeamTool(
    team=research_team,
    description="Conducts comprehensive research on topics",
    name="research_team"
)

# Use in a higher-level agent
ceo_agent = AssistantAgent(
    "ceo",
    model_client=model_client,
    tools=[research_tool]
)
```

## Creating Custom Tools

While AgentChat provides these high-level tool wrappers, you'll often use tools from `autogen_core` and `autogen_ext`:

### Basic Function Tools

```python
from autogen_core.tools import FunctionTool

# Simple synchronous tool
def calculate_compound_interest(
    principal: float,
    rate: float,
    time: int,
    n: int = 12
) -> float:
    """Calculate compound interest.
    
    Args:
        principal: Initial amount
        rate: Annual interest rate (as decimal)
        time: Time period in years
        n: Compounding frequency per year
    """
    return principal * (1 + rate/n) ** (n * time)

# Create tool with automatic schema generation
interest_tool = FunctionTool(
    calculate_compound_interest,
    description="Calculate compound interest"
)
```

### Async Tools

```python
import aiohttp
from typing import Dict, Any

async def fetch_weather(city: str, units: str = "celsius") -> Dict[str, Any]:
    """Fetch weather data for a city.
    
    Args:
        city: City name
        units: Temperature units (celsius or fahrenheit)
    """
    async with aiohttp.ClientSession() as session:
        # API call implementation
        return {"temp": 22, "condition": "sunny"}

weather_tool = FunctionTool(
    fetch_weather,
    description="Get current weather information"
)
```

### Tools with Complex Types

```python
from pydantic import BaseModel, Field
from typing import List

class SearchQuery(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum results")
    filters: List[str] = Field(default_factory=list, description="Search filters")

async def advanced_search(params: SearchQuery) -> List[Dict[str, str]]:
    """Perform advanced search with filters."""
    # Implementation
    return [{"title": "Result 1", "url": "..."}]

search_tool = FunctionTool(
    advanced_search,
    description="Advanced search with filtering"
)
```

## Tool Integration Patterns

### 1. Direct Tool Assignment

```python
agent = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[tool1, tool2, tool3]
)
```

### 2. Dynamic Tool Loading

```python
def get_tools_for_task(task_type: str) -> List[FunctionTool]:
    """Load tools based on task requirements."""
    if task_type == "research":
        return [search_tool, summarize_tool, cite_tool]
    elif task_type == "coding":
        return [lint_tool, test_tool, format_tool]
    return []

# Create agent with task-specific tools
agent = AssistantAgent(
    "dynamic",
    model_client=model_client,
    tools=get_tools_for_task(user_task_type)
)
```

### 3. Tool Composition

```python
class ToolChain:
    """Chain multiple tools together."""
    
    def __init__(self, tools: List[FunctionTool]):
        self.tools = tools
    
    async def execute(self, initial_input: Any) -> Any:
        result = initial_input
        for tool in self.tools:
            result = await tool.call_func(result)
        return result

# Create composite tool
pipeline = ToolChain([
    extract_tool,
    transform_tool,
    load_tool
])

etl_tool = FunctionTool(
    pipeline.execute,
    description="Extract, transform, and load data"
)
```

## Error Handling in Tools

### Graceful Failure

```python
async def safe_api_call(endpoint: str) -> Dict[str, Any]:
    """API call with error handling."""
    try:
        response = await make_api_request(endpoint)
        return {"success": True, "data": response}
    except HTTPError as e:
        return {"success": False, "error": f"HTTP {e.status}: {e.message}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

api_tool = FunctionTool(
    safe_api_call,
    description="Make API calls with error handling"
)
```

### Validation and Constraints

```python
from pydantic import validator

class DatabaseQuery(BaseModel):
    table: str
    limit: int = Field(default=100, le=1000)
    
    @validator('table')
    def validate_table(cls, v):
        allowed_tables = ['users', 'orders', 'products']
        if v not in allowed_tables:
            raise ValueError(f"Table must be one of {allowed_tables}")
        return v

async def query_database(params: DatabaseQuery) -> List[Dict]:
    """Safe database querying."""
    # Implementation with built-in validation
    pass
```

## Testing Tools

### Unit Testing

```python
import pytest

@pytest.mark.asyncio
async def test_weather_tool():
    # Test with mock response
    result = await weather_tool.call_func("London", "celsius")
    assert "temp" in result
    assert isinstance(result["temp"], (int, float))

def test_tool_schema():
    # Verify tool schema generation
    schema = weather_tool.schema
    assert "parameters" in schema
    assert "city" in schema["parameters"]["properties"]
```

### Integration Testing with Agents

```python
@pytest.mark.asyncio
async def test_agent_with_tools():
    # Create test agent
    agent = AssistantAgent(
        "test",
        model_client=mock_client,
        tools=[calculator_tool, weather_tool]
    )
    
    # Test tool usage
    result = await agent.run(
        task="What's the weather in Paris and calculate 15% tip on €50?"
    )
    
    # Verify both tools were called
    assert "weather" in result.messages[-1].content.lower()
    assert "7.5" in result.messages[-1].content  # 15% of 50
```

## Best Practices

### 1. Tool Design

- **Single Responsibility**: Each tool should do one thing well
- **Clear Descriptions**: Help the LLM understand when to use the tool
- **Type Hints**: Always use type hints for automatic schema generation
- **Docstrings**: Provide detailed parameter descriptions

### 2. Error Handling

- **Return Structured Errors**: Use consistent error formats
- **Graceful Degradation**: Tools should fail gracefully
- **Validation**: Validate inputs before processing
- **Timeouts**: Implement timeouts for external calls

### 3. Performance

- **Async by Default**: Use async functions for I/O operations
- **Caching**: Cache expensive operations when appropriate
- **Batch Operations**: Support batch processing where possible
- **Resource Management**: Clean up resources properly

### 4. Security

- **Input Sanitization**: Always sanitize user inputs
- **API Key Management**: Never hardcode credentials
- **Rate Limiting**: Implement rate limits for external APIs
- **Scope Limitation**: Limit tool capabilities appropriately

## Common Patterns

### Research Assistant Tools

```python
research_tools = [
    FunctionTool(web_search, description="Search the web"),
    FunctionTool(fetch_paper, description="Fetch academic papers"),
    FunctionTool(summarize_text, description="Summarize long texts"),
    FunctionTool(extract_citations, description="Extract citations")
]
```

### DevOps Tools

```python
devops_tools = [
    FunctionTool(run_tests, description="Run test suite"),
    FunctionTool(check_deployment, description="Check deployment status"),
    FunctionTool(analyze_logs, description="Analyze system logs"),
    FunctionTool(metric_query, description="Query metrics")
]
```

### Data Analysis Tools

```python
analysis_tools = [
    FunctionTool(load_dataset, description="Load data from various sources"),
    FunctionTool(statistical_analysis, description="Perform statistical analysis"),
    FunctionTool(create_visualization, description="Create charts and graphs"),
    FunctionTool(export_results, description="Export analysis results")
]
```

## Troubleshooting

### Common Issues

1. **Tool Not Being Called**
   - Check tool description clarity
   - Verify the agent's system message encourages tool use
   - Ensure the model supports function calling

2. **Schema Generation Errors**
   - Use proper type hints
   - Avoid complex nested types
   - Test schema generation separately

3. **Tool Execution Failures**
   - Implement proper error handling
   - Check input validation
   - Verify external dependencies

### Debugging Tips

```python
# Enable tool call logging
import logging
logging.getLogger("autogen_core.tools").setLevel(logging.DEBUG)

# Track tool calls
class DebugAgent(AssistantAgent):
    async def on_tool_call(self, tool_call):
        print(f"Calling tool: {tool_call.name}")
        print(f"Arguments: {tool_call.arguments}")
        result = await super().on_tool_call(tool_call)
        print(f"Result: {result}")
        return result
```

## Related Resources

- Core tools documentation: `autogen_core.tools`
- Extended tools: `autogen_ext/tools/CLAUDE.md`
- Model-specific tools: Check model client documentation
- Sample implementations: See `python/samples/` for real examples

Remember: Tools are what make your agents powerful. Choose the right tools for your use case, implement them carefully, and your agents will be able to accomplish amazing things!