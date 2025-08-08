# CLAUDE.md - AutoGen Root Documentation

*As your AutoGen guide, I use these architecture documents to help explain advanced concepts and answer deeper questions about how AutoGen works internally.*

This guide helps AutoGen users navigate the root-level documentation and understand the architecture design documents.

## Overview

The root `/docs` directory contains high-level architecture and design documents that help users understand AutoGen's internal design principles. While these are more technical than typical user documentation, they provide valuable context for advanced users who want to understand how AutoGen works under the hood.

## Documentation Structure

### Design Documents (`/design`)

These documents explain AutoGen's architectural decisions and core concepts:

1. **Programming Model** (`01 - Programming Model.md`)
   - How agents are structured and communicate
   - Event-driven architecture principles
   - Message passing patterns
   - Useful for: Understanding why AutoGen agents behave the way they do

2. **Topics** (`02 - Topics.md`)
   - Topic-based message routing system
   - Subscription mechanisms
   - How agents filter and receive messages
   - Useful for: Building complex multi-agent systems with selective communication

3. **Agent Worker Protocol** (`03 - Agent Worker Protocol.md`)
   - Distributed agent execution model
   - Cross-language communication protocol
   - Worker lifecycle management
   - Useful for: Understanding distributed deployments and scaling

4. **Agent and Topic ID Specs** (`04 - Agent and Topic ID Specs.md`)
   - Naming conventions and constraints
   - ID generation and uniqueness
   - Best practices for agent identification
   - Useful for: Properly naming agents in production systems

5. **Services** (`05 - Services.md`)
   - Service-oriented architecture concepts
   - How agents can be exposed as services
   - Integration patterns
   - Useful for: Building microservice-style agent applications

### .NET Documentation (`/dotnet`)

Contains documentation for the .NET implementation of AutoGen:

- **Core Documentation** (`/dotnet/core`)
  - Installation guide for .NET SDK
  - Tutorial for .NET developers
  - Differences from Python implementation
  - Protobuf message type definitions

- **Template and Configuration**
  - DocFX configuration for documentation generation
  - Custom styling and templates

## Key Concepts for Users

### Understanding the Architecture

While these design docs are technical, they help users understand:

1. **Why AutoGen is Event-Driven**
   - Enables scalable, distributed systems
   - Supports asynchronous communication
   - Allows loose coupling between agents

2. **How Topics Enable Flexible Communication**
   - Agents don't need direct references to each other
   - Publish-subscribe pattern for broadcasting
   - Selective message filtering

3. **Cross-Language Support**
   - Python and .NET implementations share protocols
   - Enables polyglot agent systems
   - Consistent behavior across languages

### When to Consult These Docs

Users should refer to these design documents when:

- **Scaling Beyond Single Process**: Understanding the distributed runtime
- **Building Complex Communication Patterns**: Using topics effectively
- **Integrating with Existing Systems**: Understanding the service model
- **Debugging Agent Behavior**: Understanding the underlying protocols
- **Using .NET Instead of Python**: Platform-specific guidance

## Navigation Tips

1. **Start with Programming Model** if you want to understand AutoGen's foundations
2. **Read Topics and Agent ID Specs** for multi-agent system design
3. **Consult Agent Worker Protocol** for distributed deployments
4. **Check .NET docs** if using C# or F#

## Related Documentation

- For practical Python usage: See `/python/packages/autogen-core/docs`
- For high-level agent patterns: See AgentChat documentation
- For code examples: See the cookbook and tutorials in the Python docs

Remember: These design documents provide context and rationale. For hands-on implementation, refer to the user guides and tutorials in the Python package documentation.