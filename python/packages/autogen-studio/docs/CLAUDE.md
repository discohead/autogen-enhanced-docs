# CLAUDE.md - AutoGen Studio Documentation

*As your AutoGen guide, I can help you transition from visual prototyping in Studio to production code, or show you how to maximize Studio's capabilities for your use case.*

This guide helps users understand AutoGen Studio's visual interface for building and testing multi-agent applications without writing code.

## Overview

AutoGen Studio provides a web-based UI for prototyping AutoGen applications. The documentation here is minimal as the tool is designed to be self-explanatory through its interface.

## Key Resources

### Visual Documentation

**Main Screenshot** (`ags_screen.png`)
- Shows the Studio interface layout
- Demonstrates the visual team builder
- Illustrates the testing playground

## Understanding AutoGen Studio

### What It Is
- **No-Code Agent Builder**: Create agents and teams visually
- **Testing Playground**: Interact with your agents immediately  
- **Component Gallery**: Reuse community-created components
- **Export Capability**: Generate Python code from visual designs

### Who Should Use It
- **Non-Programmers**: Business analysts, product managers
- **Rapid Prototyping**: Developers testing ideas quickly
- **Learning Tool**: Understanding agent patterns visually
- **Demo Creation**: Showing stakeholders agent capabilities

## Key Features

### 1. Visual Team Builder
- Drag-and-drop agent creation
- Connect agents to form teams
- Configure agent properties through forms
- Set up conversation flows visually

### 2. Component Gallery
- Pre-built agent templates
- Community-shared components
- Skill and tool libraries
- Model configurations

### 3. Testing Playground
- Real-time agent interaction
- Message flow visualization
- Debug conversation paths
- Test different scenarios

### 4. Code Export
- Generate Python code from visual design
- Export ready-to-run applications
- Learn AutoGen patterns by example
- Bridge from prototyping to production

## Getting Started with Studio

### Installation
See the main Studio documentation at:
`/python/packages/autogen-core/docs/src/user-guide/autogenstudio-user-guide/`

Key files:
- `installation.md` - Setup instructions
- `usage.md` - User guide
- `faq.md` - Common questions

### Typical Workflow

1. **Create Agents**
   - Choose agent type (Assistant, User, Code Executor)
   - Configure with model/tools/skills
   - Set system prompts

2. **Build Teams**  
   - Add agents to teams
   - Choose coordination pattern
   - Set termination conditions

3. **Test in Playground**
   - Send messages to teams
   - Watch agents collaborate
   - Iterate on configurations

4. **Export or Deploy**
   - Generate Python code
   - Deploy as web service
   - Share in gallery

## Studio vs Code

### When to Use Studio
- Exploring AutoGen capabilities
- Building proof-of-concepts
- Training non-technical users
- Rapid iteration on agent design

### When to Use Code
- Production deployments
- Complex custom logic
- Integration with existing systems
- Performance optimization

## Interface Components

Based on the screenshot (`ags_screen.png`):

### Navigation Panel (Left)
- **Teams**: Manage agent teams
- **Models**: Configure LLM connections
- **Agents**: Individual agent management
- **Components**: Reusable building blocks

### Main Canvas (Center)
- Visual representation of agent teams
- Drag-and-drop interface
- Connection visualization
- Real-time updates

### Properties Panel (Right)
- Agent/team configuration
- Model selection
- Tool assignment
- Prompt editing

### Testing Panel (Bottom)
- Message input
- Conversation history
- Response visualization
- Debug information

## Best Practices

### 1. Start Simple
- Begin with two-agent teams
- Use pre-built components
- Test frequently in playground

### 2. Iterate Visually
- Adjust prompts and test
- Try different team structures
- Experiment with tools

### 3. Learn Patterns
- Observe generated code
- Understand agent interactions
- Apply patterns in production

### 4. Share and Reuse
- Export successful designs
- Contribute to gallery
- Build on community work

## Limitations

### Studio is Great For:
- Prototyping and experimentation
- Learning AutoGen concepts
- Simple to medium complexity apps
- Standard agent patterns

### Studio is Limited For:
- Complex custom logic
- High-performance requirements
- Advanced integration needs
- Specialized agent behaviors

## Related Resources

### Full Documentation
- Location: `/python/packages/autogen-core/docs/src/user-guide/autogenstudio-user-guide/`
- Includes detailed setup, usage, and deployment guides

### Learning Path
1. Install Studio
2. Complete interactive tutorials
3. Build sample applications
4. Export and customize code
5. Deploy to production

### Community
- Gallery for sharing components
- Example applications
- Best practices and patterns

## Quick Tips

1. **Model Configuration**: Set up your LLM API keys in Models section first
2. **Start with Templates**: Use gallery components to learn faster
3. **Test Often**: Use the playground after each change
4. **Export Early**: Look at generated code to understand patterns
5. **Version Control**: Export and save your designs regularly

Remember: AutoGen Studio is designed for exploration and prototyping. It's the fastest way to understand AutoGen's capabilities and test ideas before implementing them in code.