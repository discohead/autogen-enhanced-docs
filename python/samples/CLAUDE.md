# CLAUDE.md - AutoGen Python Samples Guide

*I use this comprehensive sample guide to quickly find and explain relevant examples that match your specific needs - from simple demos to production-ready patterns.*

This guide provides a comprehensive map of AutoGen Python samples, helping users quickly find relevant examples for their use cases.

## Overview

The Python samples demonstrate practical implementations of AutoGen across different domains, complexity levels, and integration patterns. They range from simple UI integrations to complex distributed systems and cutting-edge research implementations.

## Sample Categories

### 1. Web UI Integration Samples

#### **agentchat_chainlit** - Interactive Chat Interface
- **What it shows**: Real-time chat UI with streaming responses
- **Key features**: 
  - Single agent and team chat interfaces
  - Tool usage integration
  - Human approval workflows
  - Round-robin group chat patterns
- **Best for**: Building conversational AI interfaces
- **Files**: `app_agent.py`, `app_team.py`, `app_team_user_proxy.py`

#### **agentchat_streamlit** - Simple Web Dashboard
- **What it shows**: Streamlit-based agent interface
- **Key features**: Basic agent interaction in web UI
- **Best for**: Quick prototypes and demos
- **Files**: `agent.py`, `main.py`

#### **agentchat_fastapi** - Production Web API
- **What it shows**: RESTful API with WebSocket support
- **Key features**:
  - State persistence
  - Conversation history management
  - WebSocket for real-time input
  - HTML frontends included
- **Best for**: Building production APIs
- **Files**: `app_agent.py`, `app_team.py`

### 2. Core Framework Samples

#### **core_async_human_in_the_loop** - Human Intervention Patterns
- **What it shows**: Async human approval workflows
- **Key features**: User intervention points in agent execution
- **Best for**: Building supervised agent systems
- **Files**: `main.py`

#### **core_chainlit** - Core API with UI
- **What it shows**: Using Core API with Chainlit
- **Key features**: Lower-level agent control with UI
- **Best for**: Advanced control with UI needs
- **Files**: `SimpleAssistantAgent.py`, `app_agent.py`, `app_team.py`

#### **core_streaming_response_fastapi** - Streaming Responses
- **What it shows**: Server-sent events for streaming
- **Key features**: Real-time response streaming
- **Best for**: Low-latency response applications
- **Files**: `app.py`

#### **core_streaming_handoffs_fastapi** - Complex Agent Handoffs
- **What it shows**: Advanced agent delegation patterns
- **Key features**:
  - Agent-to-agent handoffs
  - Chat history management
  - Tool delegation
  - Topic-based communication
- **Best for**: Complex multi-agent workflows
- **Files**: `agent_base.py`, `agent_user.py`, `app.py`, `tools.py`

### 3. Distributed & Enterprise Samples

#### **core_distributed-group-chat** - Distributed Agent System
- **What it shows**: Enterprise-scale distributed architecture
- **Key features**:
  - gRPC-based communication
  - Multiple process agents
  - Topic pub/sub patterns
  - Monitoring UI
- **Best for**: Microservices and distributed deployments
- **Files**: `run_host.py`, `run_*_agent.py`, `run_ui.py`
- **Architecture**: Host + Multiple Agent Workers + UI

#### **core_grpc_worker_runtime** - gRPC Worker Pattern
- **What it shows**: Worker-based agent distribution
- **Key features**:
  - Cascading message patterns
  - RPC and pub/sub communication
  - Worker runtime management
- **Best for**: Scalable agent deployments
- **Files**: `run_host.py`, `run_cascading_*.py`, `run_worker_*.py`

#### **core_xlang_hello_python_agent** - Cross-Language Support
- **What it shows**: Python agents with other languages
- **Key features**:
  - Protobuf definitions
  - Cross-language messaging
- **Best for**: Polyglot agent systems
- **Files**: `hello_python_agent.py`, `user_input.py`

### 4. Specialized Integration Samples

#### **agentchat_azure_postgresql** - Database Integration
- **What it shows**: PostgreSQL integration patterns
- **Key features**: Database-backed agent memory/state
- **Best for**: Persistent agent applications

#### **agentchat_graphrag** - Knowledge Graph Integration
- **What it shows**: GraphRAG for enhanced reasoning
- **Key features**:
  - Entity extraction
  - Community reports
  - Graph-based knowledge
- **Best for**: Knowledge-intensive applications
- **Files**: `app.py`, prompts directory

#### **core_semantic_router** - Intelligent Routing
- **What it shows**: Semantic message routing
- **Key features**:
  - Intent-based agent selection
  - Dynamic routing patterns
- **Best for**: Complex conversation flows
- **Files**: `_semantic_router_agent.py`, `run_semantic_router.py`

### 5. Domain-Specific Samples

#### **agentchat_chess_game** - Game Playing Agents
- **What it shows**: Turn-based game implementation
- **Key features**: Chess-playing agents
- **Best for**: Game AI patterns
- **Files**: `main.py`

#### **core_chess_game** - Core API Chess
- **What it shows**: Same chess game with Core API
- **Key features**: Lower-level control
- **Best for**: Understanding Core vs AgentChat
- **Files**: `main.py`

### 6. Real-World Applications

#### **gitty** - GitHub Automation CLI
- **What it shows**: Practical CLI tool
- **Key features**:
  - GitHub issue/PR automation
  - Draft reply generation
  - Database persistence
- **Best for**: Open source maintainers
- **Package structure**: Full Python package with `pyproject.toml`

### 7. Research & Advanced Samples

#### **task_centric_memory** - Learning & Memory Systems
- **What it shows**: Advanced memory-based learning
- **Key features**:
  - Learning from demonstrations
  - Self-teaching capabilities
  - Task-insight storage
  - Evaluation metrics
- **Best for**: AI researchers, advanced implementations
- **Files**: `eval_*.py`, configuration in `configs/`

## Sample Selection Guide

### By Use Case

| I want to... | Use this sample |
|-------------|-----------------|
| Build a chat interface | `agentchat_chainlit` |
| Create a REST API | `agentchat_fastapi` |
| Deploy distributed agents | `core_distributed-group-chat` |
| Integrate with databases | `agentchat_azure_postgresql` |
| Build a CLI tool | `gitty` |
| Add human approval | `core_async_human_in_the_loop` |
| Stream responses | `core_streaming_response_fastapi` |
| Use knowledge graphs | `agentchat_graphrag` |
| Implement learning agents | `task_centric_memory` |

### By Complexity Level

#### Beginner
- `agentchat_streamlit` - Simple UI
- `agentchat_chess_game` - Clear game logic
- `core_async_human_in_the_loop` - Basic patterns

#### Intermediate
- `agentchat_chainlit` - Interactive UI
- `agentchat_fastapi` - Web APIs
- `gitty` - Real application
- `core_semantic_router` - Smart routing

#### Advanced
- `core_distributed-group-chat` - Distributed systems
- `core_streaming_handoffs_fastapi` - Complex workflows
- `core_grpc_worker_runtime` - Worker patterns

#### Expert
- `task_centric_memory` - Research-level
- `agentchat_graphrag` - Knowledge graphs
- `core_xlang_hello_python_agent` - Cross-language

### By Integration Type

| Integration | Samples |
|------------|---------|
| Web UI | `chainlit`, `streamlit`, `fastapi` |
| Databases | `azure_postgresql` |
| APIs | `fastapi`, `gitty` (GitHub) |
| Knowledge Systems | `graphrag`, `task_centric_memory` |
| Distributed | `distributed-group-chat`, `grpc_worker` |

## Common Patterns Across Samples

### 1. Model Configuration
Most samples use `model_config.yaml` or `model_config_template.yaml`:
```yaml
provider: autogen_ext.models.openai.OpenAIChatCompletionClient
config:
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
```

### 2. Agent Setup Pattern
```python
# Common pattern seen across samples
model_client = ChatCompletionClient.load_component(config_path)
agent = AssistantAgent(name="assistant", model_client=model_client)
```

### 3. Team Coordination
- Round-robin patterns in `chainlit`
- Selector patterns in distributed samples
- Handoff patterns in `streaming_handoffs`

### 4. State Management
- In-memory (simple samples)
- Database-backed (`postgresql`, `gitty`)
- Distributed state (`distributed-group-chat`)

## Getting Started Recommendations

### For First-Time Users
1. Start with `agentchat_chainlit/app_agent.py` - simple interactive agent
2. Move to `agentchat_chainlit/app_team.py` - multi-agent coordination
3. Try `agentchat_fastapi` - build an API

### For Web Developers
1. `agentchat_fastapi` - REST APIs
2. `agentchat_chainlit` - Interactive UIs
3. `core_streaming_response_fastapi` - Advanced streaming

### For System Architects
1. `core_distributed-group-chat` - Distributed patterns
2. `core_grpc_worker_runtime` - Worker architectures
3. `core_semantic_router` - Smart routing

### For Researchers
1. `task_centric_memory` - Learning systems
2. `agentchat_graphrag` - Knowledge integration

## Running the Samples

Most samples follow this pattern:
```bash
# 1. Navigate to sample directory
cd python/samples/[sample_name]

# 2. Install dependencies
pip install -r requirements.txt  # if exists
# or
pip install autogen-agentchat autogen-ext[required_extensions]

# 3. Configure model
cp model_config_template.yaml model_config.yaml
# Edit model_config.yaml with your API keys

# 4. Run the sample
python main.py  # or app.py, or specific run_*.py files
```

## Documentation Examples

Additional examples in documentation:
- `company-research.ipynb` - Automated company research
- `literature-review.ipynb` - Academic paper analysis  
- `travel-planning.ipynb` - Multi-agent travel planning

These notebook examples in `/docs/src/user-guide/agentchat-user-guide/examples/` provide interactive learning experiences.

Remember: Each sample includes a README.md with specific setup instructions. Always check the README first for sample-specific requirements and configurations.