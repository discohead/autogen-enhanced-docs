#!/usr/bin/env python3
"""
Example usage of Claude Code integration with AutoGen.

This file demonstrates both approaches:
1. ClaudeCodeChatAgent - Full agent with native tool support
2. ClaudeCodeChatCompletionClient - Simple model client
"""

import asyncio
from pathlib import Path

# Claude Code imports
from claude_code_chat_agent import ClaudeCodeChatAgent
from claude_code_model_client import ClaudeCodeChatCompletionClient

# AutoGen imports
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SwarmGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, HandoffTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import ToolCallRequestEvent, ToolCallExecutionEvent, TextMessage
from autogen_ext.code_executors import LocalCommandLineCodeExecutor


# Example 1: Basic ClaudeCodeChatAgent usage
async def example_basic_agent():
    """Basic usage of ClaudeCodeChatAgent with tools."""
    print("\n=== Example 1: Basic ClaudeCodeChatAgent ===\n")
    
    # Create agent with file tools
    agent = ClaudeCodeChatAgent(
        name="file_assistant",
        description="A helpful assistant that can read and write files",
        system_prompt="You are a helpful coding assistant.",
        allowed_tools=["Read", "Write", "Edit"],
        permission_mode="acceptEdits",  # Auto-accept file edits
        cwd="./workspace"  # Working directory
    )
    
    # Run a task
    result = await agent.run(task="Create a hello_world.py file with a main function that prints 'Hello, World!'")
    print(f"Result: {result.messages[-1].content}")


# Example 2: Streaming with tool events
async def example_streaming_with_tools():
    """Stream responses and see tool usage events."""
    print("\n=== Example 2: Streaming with Tool Events ===\n")
    
    agent = ClaudeCodeChatAgent(
        name="analyzer",
        description="Analyzes code files",
        allowed_tools=["Read", "LS", "Grep"],
        emit_tool_events=True  # Enable tool event emission
    )
    
    # Stream and handle events
    async for event in agent.run_stream(task="List all Python files in the current directory and read the first one you find"):
        if isinstance(event, ToolCallRequestEvent):
            print(f"🔧 Tool Request: {event.content[0].name} - {event.content[0].arguments}")
        elif isinstance(event, ToolCallExecutionEvent):
            print(f"✅ Tool Executed: {event.content[0].call_id}")
        elif isinstance(event, TextMessage):
            print(f"💬 {event.source}: {event.content}")


# Example 3: Multi-agent team with code execution
async def example_team_collaboration():
    """Create a team where Claude writes code and another agent executes it."""
    print("\n=== Example 3: Multi-Agent Team Collaboration ===\n")
    
    # Create Claude Code agent for writing code
    coder = ClaudeCodeChatAgent(
        name="coder",
        description="Writes Python code",
        system_prompt="You are an expert Python developer. Write clean, well-documented code.",
        allowed_tools=["Write", "Edit"],
        permission_mode="acceptEdits",
        cwd="./workspace"
    )
    
    # Create code executor
    executor = CodeExecutorAgent(
        name="executor",
        code_executor=LocalCommandLineCodeExecutor(
            work_dir="./workspace",
            timeout=30
        )
    )
    
    # Create team
    team = RoundRobinGroupChat(
        participants=[coder, executor],
        termination_condition=MaxMessageTermination(max_messages=6)
    )
    
    # Run collaborative task
    result = await team.run(task="Create a fibonacci.py script that calculates the first 10 Fibonacci numbers, then execute it")
    
    # Print conversation
    for msg in result.messages:
        print(f"\n[{msg.source}]: {msg.content[:200]}...")


# Example 4: Swarm with handoffs
async def example_swarm_handoffs():
    """Demonstrate agent handoffs in a swarm team."""
    print("\n=== Example 4: Swarm Team with Handoffs ===\n")
    
    # Architect designs the system
    architect = ClaudeCodeChatAgent(
        name="architect",
        description="Designs software architecture",
        system_prompt="""You are a software architect. Design high-level system structure.
        When ready for implementation, say 'HANDOFF: developer' followed by implementation instructions.""",
        handoffs=["developer"]
    )
    
    # Developer implements based on design
    developer = ClaudeCodeChatAgent(
        name="developer",
        description="Implements code based on designs",
        allowed_tools=["Write", "Edit"],
        permission_mode="acceptEdits",
        system_prompt="""You implement code based on architectural designs.
        When done, say 'HANDOFF: reviewer' for code review.""",
        handoffs=["reviewer"],
        cwd="./workspace"
    )
    
    # Reviewer checks the code
    reviewer = ClaudeCodeChatAgent(
        name="reviewer",
        description="Reviews code quality",
        allowed_tools=["Read"],
        system_prompt="""You review code for quality and best practices.
        If changes needed, say 'HANDOFF: developer' with feedback.
        If approved, say 'HANDOFF: TERMINATE' to complete.""",
        handoffs=["developer", "TERMINATE"]
    )
    
    # Create swarm
    swarm = SwarmGroupChat(
        participants=[architect, developer, reviewer],
        termination_condition=HandoffTermination(target="TERMINATE")
    )
    
    # Run with streaming to see handoffs
    await Console(swarm.run_stream(task="Design and implement a simple calculator class with add and subtract methods"))


# Example 5: Simple model client usage
async def example_model_client():
    """Use ClaudeCodeChatCompletionClient for simple LLM tasks."""
    print("\n=== Example 5: Model Client for Simple Tasks ===\n")
    
    # Create model client
    model_client = ClaudeCodeChatCompletionClient(
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a helpful AI assistant.",
        # Note: Tools here won't work properly with AutoGen
    )
    
    # Use with AssistantAgent
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        model_client_stream=True
    )
    
    # Simple conversation
    await Console(agent.run_stream(task="Explain the difference between async and sync programming in Python"))
    
    # Cleanup
    await model_client.close()


# Example 6: Working directory and permissions
async def example_working_directory():
    """Demonstrate working directory and permission modes."""
    print("\n=== Example 6: Working Directory and Permissions ===\n")
    
    # Create workspace directory
    workspace = Path("./test_workspace")
    workspace.mkdir(exist_ok=True)
    
    # Strict permissions agent
    strict_agent = ClaudeCodeChatAgent(
        name="strict",
        description="Agent with strict permissions",
        allowed_tools=["Read", "Write"],
        permission_mode="default",  # Will prompt for dangerous operations
        cwd=str(workspace)
    )
    
    # Permissive agent
    permissive_agent = ClaudeCodeChatAgent(
        name="permissive",
        description="Agent with auto-accept permissions",
        allowed_tools=["Read", "Write", "Edit", "Bash"],
        permission_mode="acceptEdits",  # Auto-accept file operations
        cwd=str(workspace)
    )
    
    # Use permissive agent for file operations
    result = await permissive_agent.run(
        task="Create a config.json file with some sample configuration"
    )
    print(f"Created file: {result.messages[-1].content}")


# Example 7: State management
async def example_state_management():
    """Demonstrate saving and loading agent state."""
    print("\n=== Example 7: State Management ===\n")
    
    agent = ClaudeCodeChatAgent(
        name="stateful",
        description="Agent with state management",
        allowed_tools=["Write"]
    )
    
    # First conversation
    result1 = await agent.run(task="Create a file called notes.txt with 'Task 1 complete'")
    
    # Save state
    state = await agent.save_state()
    print(f"Saved state with {len(state['conversation_history'])} messages")
    
    # Reset agent
    await agent.on_reset(None)
    
    # Continue with new task
    result2 = await agent.run(task="What did we just do?")
    print(f"After reset: {result2.messages[-1].content}")
    
    # Restore state
    await agent.load_state(state)
    
    # Continue from saved state
    result3 = await agent.run(task="Add 'Task 2 complete' to the notes file")
    print(f"After restore: {result3.messages[-1].content}")


# Main function to run all examples
async def main():
    """Run all examples."""
    print("Claude Code + AutoGen Integration Examples")
    print("=" * 50)
    
    # Create workspace directory
    Path("./workspace").mkdir(exist_ok=True)
    
    # Run examples
    try:
        await example_basic_agent()
        await example_streaming_with_tools()
        await example_team_collaboration()
        await example_swarm_handoffs()
        await example_model_client()
        await example_working_directory()
        await example_state_management()
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())