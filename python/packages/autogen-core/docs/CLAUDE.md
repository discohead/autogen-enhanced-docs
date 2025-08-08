# CLAUDE.md - AutoGen Documentation Hub

*As your AutoGen guide, I use this documentation hub to quickly find the right resources, examples, and tutorials to help solve your specific challenges. Think of me as your personal documentation navigator.*

This guide helps AutoGen users navigate the comprehensive documentation system and find the right resources for their needs.

## Overview

This is the main documentation source for AutoGen, built with Sphinx and containing all user guides, API references, tutorials, and examples. The documentation is organized to support different user journeys - from beginners using AutoGen Studio to advanced users building with Core.

## Documentation Structure

### 1. Entry Points by User Type

#### For Beginners
- **AutoGen Studio** (`/user-guide/autogenstudio-user-guide/`)
  - Web-based UI for prototyping without code
  - Visual team builder and testing playground
  - No programming required

#### For Developers
- **AgentChat** (`/user-guide/agentchat-user-guide/`)
  - High-level Python API
  - Conversational agents and teams
  - Quick to prototype and deploy

#### For Advanced Users
- **Core** (`/user-guide/core-user-guide/`)
  - Event-driven framework
  - Maximum control and flexibility
  - Distributed agent systems

### 2. Learning Paths

#### AgentChat Learning Path
1. **Installation** (`agentchat-user-guide/installation.md`)
   - Virtual environment setup
   - Package installation
   - API key configuration

2. **Quickstart** (`agentchat-user-guide/quickstart.ipynb`)
   - First agent in minutes
   - Basic conversation example
   - Simple team coordination

3. **Tutorial Series** (`agentchat-user-guide/tutorial/`)
   - Progressive skill building:
     - `models.ipynb` - Configure LLM clients
     - `messages.ipynb` - Understand communication
     - `agents.ipynb` - Work with different agent types
     - `teams.ipynb` - Coordinate multiple agents
     - `human-in-the-loop.ipynb` - Add user interaction
     - `termination.ipynb` - Control conversations
     - `state.ipynb` - Manage agent memory

4. **Advanced Topics**
   - `custom-agents.ipynb` - Build specialized agents
   - `selector-group-chat.ipynb` - Dynamic team coordination
   - `swarm.ipynb` - Capability-based handoffs
   - `memory.ipynb` - Add persistent memory
   - `graph-flow.ipynb` - Complex workflows

5. **Examples** (`agentchat-user-guide/examples/`)
   - `travel-planning.ipynb` - Multi-agent travel assistant
   - `company-research.ipynb` - Automated research team
   - `literature-review.ipynb` - Academic paper analysis

#### Core Learning Path
1. **Quickstart** (`core-user-guide/quickstart.ipynb`)
   - Event-driven agents
   - Message passing basics

2. **Core Concepts** (`core-user-guide/core-concepts/`)
   - `agent-and-multi-agent-application.md`
   - `architecture.md`
   - `topic-and-subscription.md`
   - `agent-identity-and-lifecycle.md`

3. **Framework Guide** (`core-user-guide/framework/`)
   - `agent-and-agent-runtime.ipynb`
   - `message-and-communication.ipynb`
   - `distributed-agent-runtime.ipynb`
   - `component-config.ipynb`

4. **Design Patterns** (`core-user-guide/design-patterns/`)
   - `sequential-workflow.ipynb`
   - `group-chat.ipynb`
   - `handoffs.ipynb`
   - `concurrent-agents.ipynb`
   - `reflection.ipynb`
   - `mixture-of-agents.ipynb`

5. **Cookbook** (`core-user-guide/cookbook/`)
   - Practical recipes for common tasks
   - Integration examples
   - Production patterns

### 3. Component Guides

#### Model Clients (`components/model-clients.ipynb`)
- OpenAI, Anthropic, Azure, Gemini, Ollama
- Configuration and authentication
- Streaming and structured output

#### Tools (`components/tools.ipynb`)
- Function tools for agents
- Tool creation patterns
- Integration with external APIs

#### Code Executors (`components/command-line-code-executors.ipynb`)
- Safe code execution
- Docker and local environments
- Jupyter integration

#### Workbench (`components/workbench.ipynb`)
- Development environment
- Testing and debugging tools

### 4. Reference Documentation (`/reference/`)

Comprehensive API documentation for all packages:
- `autogen_agentchat.*` - High-level chat API
- `autogen_core.*` - Core framework
- `autogen_ext.*` - Extensions and integrations

### 5. Special Topics

#### Magentic-One (`agentchat-user-guide/magentic-one.md`)
- Microsoft's agentic system
- Pre-built specialist agents
- Complex task orchestration

#### Migration Guide (`agentchat-user-guide/migration-guide.md`)
- Upgrading from AutoGen 0.2.x
- API changes and new patterns

#### Logging and Tracing
- `logging.md` - Structured logging
- `tracing.ipynb` - OpenTelemetry integration

## Finding What You Need

### By Task

| I want to... | Start here |
|-------------|------------|
| Build my first agent | `agentchat-user-guide/quickstart.ipynb` |
| Create a multi-agent team | `agentchat-user-guide/tutorial/teams.ipynb` |
| Add tools to agents | `components/tools.ipynb` |
| Execute code safely | `components/command-line-code-executors.ipynb` |
| Use local LLMs | `cookbook/local-llms-ollama-litellm.ipynb` |
| Build distributed systems | `core-user-guide/framework/distributed-agent-runtime.ipynb` |
| Create custom agents | `agentchat-user-guide/custom-agents.ipynb` |
| Add memory to agents | `agentchat-user-guide/memory.ipynb` |

### By Integration

| Platform/Service | Documentation |
|-----------------|---------------|
| OpenAI | `components/model-clients.ipynb` |
| Azure OpenAI | `cookbook/azure-openai-with-aad-auth.md` |
| Anthropic Claude | Model clients section |
| Local LLMs (Ollama) | `cookbook/local-llms-ollama-litellm.ipynb` |
| LangChain | `cookbook/langchain-agent.ipynb` |
| LlamaIndex | `cookbook/llamaindex-agent.ipynb` |

### By Pattern

| Pattern | Example |
|---------|---------|
| Sequential workflow | `design-patterns/sequential-workflow.ipynb` |
| Parallel execution | `design-patterns/concurrent-agents.ipynb` |
| Human-in-the-loop | `tutorial/human-in-the-loop.ipynb` |
| Code generation | `design-patterns/code-execution-groupchat.ipynb` |
| Research automation | `examples/company-research.ipynb` |
| Reflection/self-improvement | `design-patterns/reflection.ipynb` |

## Visual Assets

The `/images` directory contains:
- Architecture diagrams (SVG format)
- UI screenshots
- Flow diagrams for patterns
- Product overview images

The `/drawio` directory contains editable diagrams for:
- Agent lifecycles
- Team coordination patterns
- System architecture
- Communication flows

## Documentation Features

### Interactive Notebooks
Most tutorials are Jupyter notebooks (`.ipynb`) that you can:
- Run locally with your own API keys
- Modify and experiment with
- Use as starting templates

### Visual Diagrams
Key concepts illustrated with diagrams:
- `application-stack.svg` - AutoGen component layers
- `swarm_customer_support.svg` - Swarm pattern example
- `selector-group-chat.svg` - Dynamic team selection

### Code Examples
- Every concept includes working code
- Examples progress from simple to complex
- Real-world scenarios demonstrated

## Quick Reference Card

```python
# AutoGen Hello World
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 1. Create model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# 2. Create agent  
agent = AssistantAgent("assistant", model_client=model_client)

# 3. Run task
result = await agent.run(task="Hello! Tell me a joke.")

# 4. Multi-agent team
from autogen_agentchat.teams import RoundRobinGroupChat

team = RoundRobinGroupChat([agent1, agent2])
result = await team.run(task="Collaborate on this")
```

## Getting Help

1. **Start with Quickstart** for hands-on introduction
2. **Follow the Tutorials** for structured learning
3. **Check Examples** for similar use cases
4. **Consult Cookbook** for specific integrations
5. **Read Design Patterns** for architectural guidance

Remember: The documentation is designed for learning by doing. Most pages include runnable code you can experiment with immediately.