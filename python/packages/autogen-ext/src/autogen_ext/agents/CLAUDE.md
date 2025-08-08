# CLAUDE.md - AutoGen Extended Agents Guide

*As your AutoGen mentor, I'll help you understand these specialized agents that extend AutoGen's capabilities into specific domains like web browsing, file navigation, and platform integrations.*

This guide covers the extended agents available in the `autogen-ext` package, providing specialized capabilities beyond the core agent types.

## Overview

Extended agents in AutoGen provide domain-specific capabilities while maintaining compatibility with the standard agent interface. These agents are designed for specific tasks and integrate with external services or specialized tools.

## Azure AI Agents

### AzureAIAgent

Integrates with Azure AI services for enhanced capabilities.

```python
from autogen_ext.agents.azure import AzureAIAgent

# Create Azure AI agent
azure_agent = AzureAIAgent(
    name="azure_assistant",
    azure_endpoint="https://your-resource.cognitiveservices.azure.com/",
    api_key="your-api-key",
    deployment_name="your-deployment"
)

# Use in a team like any other agent
team = RoundRobinGroupChat([azure_agent, local_agent])
```

**Key Features**:
- Azure Cognitive Services integration
- Enhanced language understanding
- Azure-specific optimizations
- Managed scaling and reliability

**When to Use**:
- Enterprise Azure deployments
- Need for Azure-specific features
- Compliance requirements
- Integration with Azure ecosystem

## OpenAI Agents

### OpenAIAgent

Direct integration with OpenAI's API, optimized for their models.

```python
from autogen_ext.agents.openai import OpenAIAgent

# Standard OpenAI agent
openai_agent = OpenAIAgent(
    name="gpt4_agent",
    model="gpt-4-turbo",
    api_key="sk-...",  # Or use OPENAI_API_KEY env var
    temperature=0.7
)
```

### OpenAIAssistantAgent

Integrates with OpenAI's Assistants API for persistent, stateful agents.

```python
from autogen_ext.agents.openai import OpenAIAssistantAgent

# Create or connect to an OpenAI Assistant
assistant_agent = OpenAIAssistantAgent(
    name="persistent_assistant",
    assistant_id="asst_...",  # Existing assistant
    # Or create new:
    instructions="You are a helpful data analyst...",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-turbo"
)

# Maintains conversation state across sessions
result = await assistant_agent.run(task="Analyze this dataset")
```

**Key Features**:
- Persistent conversation threads
- Built-in code interpreter
- File handling capabilities
- Retrieval augmentation
- Function calling

**When to Use**:
- Need persistent agent state
- Complex data analysis tasks
- File processing workflows
- Long-running conversations

## Web Surfer Agents

### MultimodalWebSurfer

A powerful agent that can browse the web, understand page content, and interact with web interfaces.

```python
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.web_surfer import PlaywrightController

# Initialize browser controller
browser_controller = PlaywrightController(
    headless=True,  # Run in background
    viewport_size=(1280, 720)
)

# Create web surfing agent
web_surfer = MultimodalWebSurfer(
    name="web_researcher",
    browser_controller=browser_controller,
    model_client=model_client,
    system_message="You are a web research specialist."
)

# Use for web tasks
result = await web_surfer.run(
    task="Go to example.com and find their pricing information"
)
```

**Advanced Configuration**:

```python
# With screenshot analysis
web_surfer = MultimodalWebSurfer(
    name="visual_browser",
    browser_controller=browser_controller,
    model_client=multimodal_model_client,  # Needs vision capabilities
    use_screenshots=True,
    use_set_of_mark=True  # Visual element identification
)

# Custom browser settings
browser_controller = PlaywrightController(
    headless=False,  # Show browser window
    viewport_size=(1920, 1080),
    user_agent="Custom Agent String",
    timeout=30000,  # 30 seconds
    cookies=[{"name": "session", "value": "..."}]
)
```

**Key Features**:
- Full web browsing capabilities
- JavaScript execution
- Form filling and interaction
- Screenshot analysis
- Element identification (Set-of-Mark)
- Cookie and session management

**When to Use**:
- Web scraping and research
- Automated web testing
- Data extraction from websites
- Web-based workflows
- Competitive analysis

## File Surfer Agents

### FileSurfer

Navigate and analyze file systems and documents.

```python
from autogen_ext.agents.file_surfer import FileSurfer

# Create file navigation agent
file_surfer = FileSurfer(
    name="document_analyzer",
    model_client=model_client,
    root_directory="/path/to/documents",  # Restrict access
    allowed_extensions=[".txt", ".md", ".pdf", ".docx"],
    max_file_size_mb=10
)

# Navigate and analyze files
result = await file_surfer.run(
    task="Find all markdown files mentioning 'API' and summarize them"
)
```

### MarkdownFileBrowser

Specialized for markdown documentation navigation.

```python
from autogen_ext.agents.file_surfer import MarkdownFileBrowser

# Create markdown specialist
md_browser = MarkdownFileBrowser(
    name="docs_expert",
    model_client=model_client,
    docs_directory="/path/to/documentation",
    create_index=True,  # Build searchable index
    follow_links=True   # Navigate between linked docs
)

# Use for documentation tasks
result = await md_browser.run(
    task="Explain how authentication works based on the docs"
)
```

**Key Features**:
- File system navigation
- Content extraction and analysis
- Multiple format support
- Security restrictions
- Efficient file indexing
- Link following in documents

**When to Use**:
- Documentation analysis
- Code repository exploration
- File-based data extraction
- Knowledge base navigation
- Automated file processing

## Video Surfer Agents

### VideoSurfer

Analyzes video content using multimodal capabilities.

```python
from autogen_ext.agents.video_surfer import VideoSurfer

# Create video analysis agent
video_surfer = VideoSurfer(
    name="video_analyst",
    model_client=multimodal_model_client,  # Needs vision
    frame_extraction_rate=1.0,  # 1 frame per second
    max_frames=100,
    include_audio_transcript=True
)

# Analyze video content
result = await video_surfer.run(
    task="Analyze video.mp4 and identify key topics discussed"
)
```

**Key Features**:
- Video frame extraction
- Scene analysis
- Audio transcription integration
- Temporal understanding
- Content summarization

**When to Use**:
- Video content analysis
- Tutorial comprehension
- Meeting recording analysis
- Content moderation
- Educational video processing

## MagenticOne Agents

### MagenticOneCoderAgent

Specialized coding agent with enhanced capabilities.

```python
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent

# Create advanced coding agent
coder = MagenticOneCoderAgent(
    name="expert_coder",
    model_client=model_client,
    languages=["python", "javascript", "rust"],
    enable_debugging=True,
    enable_testing=True,
    code_execution_config={
        "timeout": 30,
        "max_memory_mb": 512
    }
)

# Use for complex coding tasks
result = await coder.run(
    task="Implement a rate limiter with token bucket algorithm"
)
```

**Key Features**:
- Multi-language support
- Integrated debugging
- Test generation
- Code execution
- Performance optimization

**When to Use**:
- Complex coding tasks
- Multi-language projects
- Code review and optimization
- Automated testing
- Algorithm implementation

## Integration Patterns

### Combining Extended Agents

```python
# Research team with specialized agents
research_team = RoundRobinGroupChat([
    web_surfer,      # Gather online information
    file_surfer,     # Analyze local documents
    video_surfer,    # Process video content
    analyst_agent    # Synthesize findings
])

# Development team
dev_team = SelectorGroupChat(
    participants=[
        coder_agent,           # Write code
        web_surfer,           # Research solutions
        openai_assistant,     # Code review with interpreter
    ],
    model_client=model_client
)
```

### Agent Handoffs

```python
# Web research hands off to file analysis
web_surfer = MultimodalWebSurfer(
    name="web_researcher",
    handoffs=[
        Handoff(
            target="file_analyzer",
            description="Hand off downloaded files for analysis"
        )
    ]
)

file_analyzer = FileSurfer(
    name="file_analyzer",
    handoffs=[
        Handoff(
            target="report_writer",
            description="Send analysis for report generation"
        )
    ]
)
```

## Best Practices

### 1. Resource Management

```python
# Always clean up browser resources
async with browser_controller:
    web_surfer = MultimodalWebSurfer(
        name="browser",
        browser_controller=browser_controller
    )
    await team.run(task="Research task")
# Browser automatically closed

# Limit file access scope
file_surfer = FileSurfer(
    root_directory="/safe/directory",
    max_file_size_mb=10,
    allowed_extensions=[".txt", ".md"]
)
```

### 2. Error Handling

```python
# Wrap specialized operations
try:
    result = await web_surfer.run(
        task="Navigate to example.com"
    )
except NavigationError as e:
    # Handle browser-specific errors
    fallback_agent = create_fallback_agent()
    result = await fallback_agent.run(task="Research example.com via API")
```

### 3. Performance Optimization

```python
# Optimize video processing
video_surfer = VideoSurfer(
    frame_extraction_rate=0.5,  # Lower rate for long videos
    max_frames=50,              # Limit total frames
    enable_caching=True         # Cache processed frames
)

# Optimize web browsing
web_surfer = MultimodalWebSurfer(
    use_screenshots=False,      # Faster without screenshots
    wait_for_network_idle=False # Don't wait for all resources
)
```

### 4. Security Considerations

```python
# Restrict web access
browser_controller = PlaywrightController(
    blocked_urls=["*.internal.com", "localhost:*"],
    allowed_domains=["example.com", "docs.example.com"]
)

# Sandboxed file access
file_surfer = FileSurfer(
    root_directory="/sandbox",
    prevent_directory_traversal=True,
    read_only=True
)
```

## Troubleshooting

### Common Issues

1. **Browser Launch Failures**
   ```python
   # Install playwright browsers first
   # Run: playwright install chromium
   ```

2. **Model Compatibility**
   ```python
   # Ensure multimodal model for visual agents
   if not model_client.model_info.supports_vision:
       raise ValueError("Visual agents need multimodal models")
   ```

3. **Resource Limits**
   ```python
   # Monitor resource usage
   import psutil
   
   if psutil.virtual_memory().percent > 80:
       # Reduce concurrent agents
       pass
   ```

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.getLogger("autogen_ext.agents").setLevel(logging.DEBUG)

# Use non-headless browser for debugging
browser_controller = PlaywrightController(
    headless=False,
    slow_mo=1000  # Slow down actions by 1 second
)

# Save intermediate results
web_surfer = MultimodalWebSurfer(
    save_screenshots_to="/tmp/debug_shots",
    log_browser_console=True
)
```

## Performance Benchmarks

| Agent Type | Typical Response Time | Resource Usage |
|------------|---------------------|----------------|
| OpenAIAgent | 1-3 seconds | Low |
| WebSurfer | 5-20 seconds | Medium-High |
| FileSurfer | 0.5-5 seconds | Low-Medium |
| VideoSurfer | 10-60 seconds | High |
| MagenticOneCoder | 3-10 seconds | Medium |

## Related Resources

- Web automation tools: `autogen_ext.agents.web_surfer`
- File processing tools: `autogen_ext.agents.file_surfer`
- Code execution: `autogen_ext.code_executors`
- Model clients: `autogen_ext.models`

Remember: Extended agents are powerful but resource-intensive. Choose the right agent for your specific needs and always implement proper resource management!