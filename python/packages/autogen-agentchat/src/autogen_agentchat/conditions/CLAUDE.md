# CLAUDE.md - AutoGen Termination Patterns Guide

*As your AutoGen guide, I'll help you master termination conditions - the control mechanisms that prevent runaway conversations and ensure your multi-agent systems complete their tasks efficiently.*

This guide covers all termination conditions in AutoGen AgentChat, helping you control when agent conversations should end.

## Overview

Termination conditions are crucial for managing multi-agent conversations. They determine when a team should stop processing, preventing infinite loops and ensuring efficient task completion. AutoGen provides a rich set of built-in conditions that can be combined using logical operators.

## Built-in Termination Conditions

### 1. MaxMessageTermination

Stops after a specific number of messages.

```python
from autogen_agentchat.conditions import MaxMessageTermination

# Stop after 10 messages
termination = MaxMessageTermination(max_messages=10)

# Use in a team
team = RoundRobinGroupChat(
    participants=[agent1, agent2],
    termination_condition=termination
)
```

**When to use**: 
- Preventing runaway conversations
- Setting hard limits for demos
- Ensuring predictable execution time

### 2. TextMentionTermination

Stops when specific text appears in any message.

```python
from autogen_agentchat.conditions import TextMentionTermination

# Stop when "TASK COMPLETE" appears
termination = TextMentionTermination(text="TASK COMPLETE")

# Case-insensitive matching
termination = TextMentionTermination(
    text="done",
    case_sensitive=False
)
```

**When to use**:
- Task completion markers
- Error detection ("ERROR", "FAILED")
- User-defined stop words

### 3. StopMessageTermination

Stops when a `StopMessage` is sent by any agent.

```python
from autogen_agentchat.conditions import StopMessageTermination
from autogen_agentchat.messages import StopMessage

# Configure termination
termination = StopMessageTermination()

# Agent can send StopMessage
async def process_task(agent, task):
    if task_completed:
        return StopMessage(content="Task completed successfully")
```

**When to use**:
- Programmatic completion
- Clean shutdown patterns
- Integration with agent logic

### 4. TokenUsageTermination

Stops when token usage exceeds limits.

```python
from autogen_agentchat.conditions import TokenUsageTermination

# Stop at 4000 tokens total
termination = TokenUsageTermination(max_total_tokens=4000)

# Separate limits for prompt and completion
termination = TokenUsageTermination(
    max_prompt_tokens=3000,
    max_completion_tokens=1000
)
```

**When to use**:
- Cost control
- Staying within model limits
- Managing context window size

### 5. HandoffTermination

Stops when an agent performs a handoff.

```python
from autogen_agentchat.conditions import HandoffTermination
from autogen_agentchat.base import Handoff

# Configure termination
termination = HandoffTermination(target="human")

# Or terminate on any handoff
termination = HandoffTermination()

# Agent performs handoff
handoff = Handoff(target="support_team", message="Escalating to support")
```

**When to use**:
- Human-in-the-loop workflows
- Task delegation boundaries
- Escalation scenarios

### 6. TimeoutTermination

Stops after a time limit is reached.

```python
from autogen_agentchat.conditions import TimeoutTermination

# 5 minute timeout
termination = TimeoutTermination(timeout_seconds=300)

# Combine with message limit for safety
termination = TimeoutTermination(300) | MaxMessageTermination(100)
```

**When to use**:
- Real-time systems
- SLA enforcement
- Preventing hanging operations

### 7. ExternalTermination

Stops based on external signals.

```python
from autogen_agentchat.conditions import ExternalTermination

# Create termination with external control
termination = ExternalTermination()

# In another part of your application
async def monitor_system():
    if critical_event_detected():
        await termination.set()  # Triggers termination

# Reset for reuse
await termination.reset()
```

**When to use**:
- User cancellation
- System shutdown
- External event handling

### 8. SourceMatchTermination

Stops when a message from a specific source matches criteria.

```python
from autogen_agentchat.conditions import SourceMatchTermination

# Stop when the user agent says "exit"
termination = SourceMatchTermination(
    source="user",
    text="exit",
    case_sensitive=False
)
```

**When to use**:
- User-controlled termination
- Specific agent completion signals
- Role-based termination

### 9. TextMessageTermination

Stops when any text message matches a pattern.

```python
from autogen_agentchat.conditions import TextMessageTermination

# Stop on any text message containing a pattern
termination = TextMessageTermination(
    text_pattern=r"^FINAL ANSWER:",
    regex=True
)
```

**When to use**:
- Pattern-based completion
- Structured output detection
- Complex text matching

### 10. FunctionCallTermination

Stops when a specific function is called.

```python
from autogen_agentchat.conditions import FunctionCallTermination

# Stop when save_results function is called
termination = FunctionCallTermination(function_name="save_results")

# Stop on any function call
termination = FunctionCallTermination()
```

**When to use**:
- Tool-based completion
- Workflow milestones
- Action-triggered termination

### 11. FunctionalTermination

Stops based on custom function logic.

```python
from autogen_agentchat.conditions import FunctionalTermination

# Custom termination logic
async def should_terminate(messages):
    # Analyze conversation history
    if len(messages) > 5:
        last_messages = messages[-5:]
        # Check for repetition
        if len(set(m.content for m in last_messages)) == 1:
            return True
    return False

termination = FunctionalTermination(should_terminate)
```

**When to use**:
- Complex custom logic
- State-based termination
- Advanced patterns

## Combining Conditions

### Logical OR (Any condition)

```python
# Stop on timeout OR max messages
termination = TimeoutTermination(300) | MaxMessageTermination(50)

# Multiple OR conditions
termination = (
    TextMentionTermination("DONE") |
    StopMessageTermination() |
    HandoffTermination()
)
```

### Logical AND (All conditions)

```python
# Stop only when BOTH conditions are met
termination = (
    TextMentionTermination("APPROVED") &
    SourceMatchTermination(source="manager")
)
```

### Complex Combinations

```python
# (Timeout OR max messages) AND (completion marker OR error)
safety_limit = TimeoutTermination(300) | MaxMessageTermination(100)
completion = TextMentionTermination("COMPLETE") | TextMentionTermination("ERROR")
termination = safety_limit & completion
```

## Custom Termination Conditions

### Creating Your Own

```python
from autogen_agentchat.conditions import TerminationCondition
from autogen_agentchat.messages import ChatMessage
from typing import List

class ConsensusTermination(TerminationCondition):
    """Terminate when all agents agree."""
    
    def __init__(self, agreement_phrase: str = "I AGREE"):
        self._agreement_phrase = agreement_phrase
        self._agreements = set()
    
    async def is_terminal(self, messages: List[ChatMessage]) -> bool:
        if not messages:
            return False
            
        last_message = messages[-1]
        
        # Track agreements
        if self._agreement_phrase in last_message.content:
            self._agreements.add(last_message.source)
        
        # Check if all participants agree
        participants = {msg.source for msg in messages}
        return self._agreements == participants
    
    async def reset(self) -> None:
        self._agreements.clear()
```

### Stateful Termination

```python
class QualityTermination(TerminationCondition):
    """Terminate when output quality is sufficient."""
    
    def __init__(self, quality_checker, threshold: float = 0.8):
        self._checker = quality_checker
        self._threshold = threshold
    
    async def is_terminal(self, messages: List[ChatMessage]) -> bool:
        if len(messages) < 2:
            return False
            
        # Check quality of last assistant message
        for msg in reversed(messages):
            if msg.source == "assistant":
                score = await self._checker.evaluate(msg.content)
                return score >= self._threshold
        
        return False
```

## Best Practices

### 1. Always Set Termination

```python
# Bad: No termination (can run forever)
team = RoundRobinGroupChat([agent1, agent2])

# Good: Multiple safety conditions
termination = (
    MaxMessageTermination(50) |
    TimeoutTermination(600) |
    TextMentionTermination("COMPLETE")
)
team = RoundRobinGroupChat([agent1, agent2], termination_condition=termination)
```

### 2. Combine Safety and Logic Conditions

```python
# Safety conditions (prevent runaway)
safety = MaxMessageTermination(100) | TimeoutTermination(300)

# Logic conditions (task completion)
completion = TextMentionTermination("DONE") | HandoffTermination()

# Combined
termination = safety | completion
```

### 3. Clear Completion Signals

```python
# Configure agents to emit clear signals
agent = AssistantAgent(
    "assistant",
    model_client=model_client,
    system_message="""
    You are a helpful assistant.
    When you have completed the task, always end your message with "TASK COMPLETE".
    """
)

# Match the signal
termination = TextMentionTermination("TASK COMPLETE")
```

### 4. Handle Edge Cases

```python
# Prevent termination on first message
class DelayedTermination(TerminationCondition):
    def __init__(self, base_condition, min_messages: int = 2):
        self._base = base_condition
        self._min_messages = min_messages
    
    async def is_terminal(self, messages):
        if len(messages) < self._min_messages:
            return False
        return await self._base.is_terminal(messages)
```

## Common Patterns

### Research Task Pattern

```python
# Research continues until findings are complete
research_termination = (
    TextMentionTermination("RESEARCH COMPLETE") |
    TextMentionTermination("NO MORE RESULTS") |
    MaxMessageTermination(30) |
    TimeoutTermination(600)
)
```

### Interactive Session Pattern

```python
# User-controlled session
interactive_termination = (
    SourceMatchTermination(source="user", text="exit") |
    SourceMatchTermination(source="user", text="quit") |
    TimeoutTermination(1800)  # 30 min timeout
)
```

### Quality Assurance Pattern

```python
# Continue until quality threshold met
qa_termination = (
    FunctionalTermination(check_quality_threshold) |
    MaxMessageTermination(20) |
    TextMentionTermination("QUALITY APPROVED")
)
```

### Error Handling Pattern

```python
# Stop on any error
error_termination = (
    TextMentionTermination("ERROR") |
    TextMentionTermination("EXCEPTION") |
    TextMentionTermination("FAILED") |
    MaxMessageTermination(50)
)
```

## Debugging Termination

### Logging Termination Reasons

```python
import logging

class LoggingTermination(TerminationCondition):
    def __init__(self, base_condition, name: str):
        self._base = base_condition
        self._name = name
        self._logger = logging.getLogger(__name__)
    
    async def is_terminal(self, messages):
        result = await self._base.is_terminal(messages)
        if result:
            self._logger.info(f"Termination triggered by: {self._name}")
        return result

# Wrap conditions with logging
termination = LoggingTermination(
    MaxMessageTermination(10),
    "max_messages"
) | LoggingTermination(
    TextMentionTermination("DONE"),
    "completion_marker"
)
```

### Testing Termination

```python
import pytest

@pytest.mark.asyncio
async def test_termination_condition():
    termination = TextMentionTermination("STOP")
    
    messages = [
        ChatMessage(content="Hello", source="user"),
        ChatMessage(content="Working...", source="assistant")
    ]
    
    # Should not terminate
    assert not await termination.is_terminal(messages)
    
    # Add termination message
    messages.append(ChatMessage(content="STOP now", source="user"))
    
    # Should terminate
    assert await termination.is_terminal(messages)
```

## Performance Considerations

1. **Efficient Conditions First**: Put cheap checks before expensive ones
2. **Avoid Heavy Computation**: Keep termination checks lightweight
3. **Cache Results**: For expensive checks, cache results when possible
4. **Reset State**: Always implement proper reset for reusable conditions

## Troubleshooting

### Common Issues

1. **Conversation Never Ends**
   - Add safety conditions (MaxMessage, Timeout)
   - Check if termination text matches exactly
   - Verify condition logic (AND vs OR)

2. **Premature Termination**
   - Check for overly broad text matching
   - Ensure minimum message requirements
   - Review condition combinations

3. **Inconsistent Behavior**
   - Reset stateful conditions between runs
   - Check for race conditions in async code
   - Verify message source names match

Remember: Good termination conditions are essential for reliable multi-agent systems. Always combine safety limits with task-specific conditions!