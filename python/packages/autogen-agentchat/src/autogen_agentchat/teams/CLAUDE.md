# CLAUDE.md - AutoGen AgentChat Teams

*I'll help you choose the right team pattern and show you how to orchestrate agents effectively for your specific application needs.*

This guide provides detailed information about team patterns in AutoGen for coordinating multiple agents.

## Overview

Teams in AutoGen orchestrate how multiple agents work together. Each team type implements a different coordination pattern suitable for various multi-agent scenarios.

## Team Types

### RoundRobinGroupChat

Agents take turns in a fixed sequential order.

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

# Agents speak in order: agent1 -> agent2 -> agent3 -> agent1...
team = RoundRobinGroupChat(
    participants=[agent1, agent2, agent3],
    termination_condition=MaxMessageTermination(max_messages=10),
    max_turns=None  # No turn limit
)

# Run the team
result = await team.run(task="Collaborate on this task")
```

#### When to Use
- Predictable workflows (e.g., research → analysis → summary)
- Fair distribution of participation
- Testing and debugging multi-agent flows

#### Advanced Configuration
```python
team = RoundRobinGroupChat(
    participants=[researcher, analyst, writer],
    termination_condition=MaxMessageTermination(9) | TextMentionTermination("DONE"),
    max_turns=3,  # Each agent gets at most 3 turns
    emit_team_events=True  # Enable detailed event tracking
)
```

### SelectorGroupChat

Uses an LLM to dynamically select the next speaker based on conversation context.

```python
from autogen_agentchat.teams import SelectorGroupChat

team = SelectorGroupChat(
    participants=[expert1, expert2, expert3],
    model_client=model_client,
    termination_condition=TextMentionTermination("TERMINATE"),
    selector_prompt="""You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role.
"""
)
```

#### Custom Selection Logic
```python
def custom_selector(messages):
    """Return agent name based on conversation state"""
    last_message = messages[-1].content if messages else ""
    
    if "error" in last_message.lower():
        return "debugger"
    elif "analyze" in last_message.lower():
        return "analyst"
    return None  # Let model decide

team = SelectorGroupChat(
    participants=[debugger, analyst, designer],
    model_client=model_client,
    selector_func=custom_selector,  # Override model selection
    allow_repeated_speaker=False,  # Prevent same agent twice in a row
    max_selector_attempts=3  # Retry selection if invalid
)
```

#### Model Context for Selection
```python
from autogen_core.model_context import BufferedChatCompletionContext

# Limit context for speaker selection
selection_context = BufferedChatCompletionContext(buffer_size=5)

team = SelectorGroupChat(
    participants=[agent1, agent2, agent3],
    model_client=model_client,
    model_context=selection_context,
    model_client_streaming=True  # Stream selection reasoning
)
```

### SwarmGroupChat

Agents can hand off control to other agents based on their capabilities.

```python
from autogen_agentchat.teams import SwarmGroupChat
from autogen_agentchat.base import Handoff

# Define handoff patterns
agent1 = AssistantAgent(
    "coordinator",
    model_client=model_client,
    handoffs=[
        Handoff(target="specialist", message="Transfer to specialist for technical work"),
        Handoff(target="reviewer", message="Transfer to reviewer for quality check")
    ]
)

team = SwarmGroupChat(
    participants=[agent1, specialist, reviewer],
    termination_condition=HandoffTermination(target="TERMINATE")
)
```

#### Dynamic Handoffs
```python
# Agents decide handoffs based on context
specialist = AssistantAgent(
    "specialist",
    model_client=model_client,
    system_message="You handle technical implementation. Hand off to 'reviewer' when complete.",
    handoffs=[
        Handoff(target="reviewer", message="Implementation complete, please review"),
        Handoff(target="coordinator", message="Need clarification on requirements")
    ]
)
```

## Termination Conditions

### Built-in Conditions

```python
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    HandoffTermination,
    TimeoutTermination,
    ExternalStopTermination,
    StopMessageTermination
)

# Single condition
termination = MaxMessageTermination(max_messages=20)

# Combined conditions (OR logic)
termination = (
    MaxMessageTermination(30) |
    TextMentionTermination("DONE") |
    TimeoutTermination(timeout_seconds=300)
)

# Multiple text patterns
termination = TextMentionTermination(
    texts=["TERMINATE", "DONE", "STOP"],
    sources=["coordinator", "reviewer"]  # Only these agents can terminate
)
```

### Custom Termination

```python
from autogen_agentchat.conditions import TerminationCondition

class QualityTermination(TerminationCondition):
    def __init__(self, quality_threshold: float):
        self.threshold = quality_threshold
        self._scores = []
    
    async def is_terminal(self, messages) -> bool:
        # Check if quality threshold met
        if "quality_score:" in messages[-1].content:
            score = float(messages[-1].content.split("quality_score:")[1])
            self._scores.append(score)
            return score >= self.threshold
        return False
    
    async def reset(self) -> None:
        self._scores.clear()

team = RoundRobinGroupChat(
    participants=[writer, reviewer],
    termination_condition=QualityTermination(0.9)
)
```

## Team Execution Patterns

### Streaming Execution
```python
from autogen_agentchat.ui import Console

# Stream team execution to console
await Console(team.run_stream(task="Your task"))

# Custom streaming handler
async for message in team.run_stream(task="Your task"):
    if isinstance(message, TextMessage):
        print(f"{message.source}: {message.content}")
    elif isinstance(message, ToolCallRequestEvent):
        print(f"Calling tool: {message.tool_call.name}")
```

### Batch Execution
```python
# Run multiple tasks
tasks = ["Task 1", "Task 2", "Task 3"]
results = []

for task in tasks:
    await team.reset()  # Clear state between tasks
    result = await team.run(task=task)
    results.append(result)
```

### State Management
```python
# Save team state
state = await team.save_state()

# Continue conversation later
await team.load_state(state)
result = await team.run(task="Continue from where we left off")
```

## Advanced Patterns

### Nested Teams
```python
# Team of teams
research_team = RoundRobinGroupChat([researcher1, researcher2])
analysis_team = SelectorGroupChat([analyst1, analyst2], model_client)

# Create agents that represent teams
research_leader = SocietyOfMindAgent("research_lead", team=research_team)
analysis_leader = SocietyOfMindAgent("analysis_lead", team=analysis_team)

# Top-level coordination
main_team = SwarmGroupChat([coordinator, research_leader, analysis_leader])
```

### Dynamic Team Composition
```python
class AdaptiveTeam:
    def __init__(self, agent_pool, model_client):
        self.agent_pool = agent_pool
        self.model_client = model_client
    
    async def run(self, task):
        # Select agents based on task
        selected_agents = await self._select_agents(task)
        
        # Create appropriate team type
        if len(selected_agents) == 2:
            team = RoundRobinGroupChat(selected_agents)
        else:
            team = SelectorGroupChat(selected_agents, self.model_client)
        
        return await team.run(task=task)
```

### Event Handling
```python
# Enable team events
team = SelectorGroupChat(
    participants=[agent1, agent2],
    model_client=model_client,
    emit_team_events=True
)

# Process events during streaming
async for event in team.run_stream(task="Task"):
    if isinstance(event, SelectorEvent):
        print(f"Selected: {event.content}")
    elif isinstance(event, ModelClientStreamingChunkEvent):
        print(event.content, end="")
```

## Performance Optimization

### Context Management
```python
# Limit context size for large conversations
from autogen_core.model_context import BufferedChatCompletionContext

# Each agent maintains limited context
for agent in [agent1, agent2, agent3]:
    agent._model_context = BufferedChatCompletionContext(buffer_size=10)

team = SelectorGroupChat(
    participants=[agent1, agent2, agent3],
    model_client=model_client,
    model_context=BufferedChatCompletionContext(buffer_size=5)  # Selection context
)
```

### Parallel Processing
```python
# Run independent subtasks in parallel
async def parallel_research(topics):
    teams = [
        RoundRobinGroupChat([researcher, summarizer])
        for _ in topics
    ]
    
    tasks = [
        team.run(task=f"Research: {topic}")
        for team, topic in zip(teams, topics)
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

## Error Handling

```python
from autogen_agentchat.base import TeamRunException

try:
    result = await team.run(task="Complex task")
except TeamRunException as e:
    print(f"Team execution failed: {e}")
    # Access partial results
    if e.partial_result:
        print(f"Completed {len(e.partial_result.messages)} messages")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Testing Teams

```python
from autogen_ext.models.replay import ReplayChatCompletionClient

# Deterministic team behavior
test_client = ReplayChatCompletionClient(
    responses=["Response 1", "agent2", "Response 2", "DONE"]
)

# Test agents
agent1 = AssistantAgent("agent1", model_client=test_client)
agent2 = AssistantAgent("agent2", model_client=test_client)

# Test team
team = SelectorGroupChat(
    participants=[agent1, agent2],
    model_client=test_client,
    termination_condition=TextMentionTermination("DONE")
)

result = await team.run(task="Test")
assert len(result.messages) == 4
```

## Best Practices

1. **Choose the right team type**:
   - RoundRobin: Predictable workflows
   - Selector: Dynamic expertise selection
   - Swarm: Capability-based delegation

2. **Set clear termination conditions** to prevent infinite loops

3. **Use descriptive agent names and descriptions** for better selection

4. **Monitor team execution** with events and streaming

5. **Test team behavior** with deterministic model clients

6. **Handle errors gracefully** at the team level

7. **Optimize context size** for long conversations

Remember: Teams are composable - you can nest teams within teams for complex hierarchical workflows.