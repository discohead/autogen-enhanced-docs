# AutoGen Teams Mastery Guide: Multi-Agent Coordination Patterns

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Implementation Guide](#implementation-guide)
3. [Advanced Patterns](#advanced-patterns)
4. [Practical Examples](#practical-examples)
5. [Optimization & Debugging](#optimization--debugging)

---

## Fundamentals

### What Are Teams?

Teams in AutoGen are orchestrated groups of agents that collaborate to solve complex problems. Think of them as specialized task forces where each agent brings unique capabilities, and the team coordination pattern determines how they interact.

### Why Use Teams?

Teams excel when:
- **Tasks require diverse expertise**: Different agents handle different aspects (research, analysis, coding, review)
- **Problems need structured workflows**: Sequential processing, parallel execution, or dynamic delegation
- **Quality requires multiple perspectives**: Reflection patterns, peer review, consensus building
- **Scale demands distribution**: Breaking complex problems into manageable sub-tasks

### Available Team Types

AutoGen provides five main team coordination patterns, each optimized for different scenarios:

| Team Type | Coordination Pattern | Best For | Key Feature |
|-----------|---------------------|----------|-------------|
| **RoundRobinGroupChat** | Sequential turns | Equal participation, predictable flow | Agents speak in fixed order |
| **SelectorGroupChat** | AI-driven selection | Dynamic conversations | Model selects next speaker |
| **SwarmGroupChat** | Explicit handoffs | Workflow orchestration | Agents pass control explicitly |
| **MagenticOneGroupChat** | Intelligent orchestration | Complex problem-solving | Progress tracking & re-planning |
| **GraphFlow** | Graph-based execution | Complex workflows | Branching, loops, parallel execution |

---

## Implementation Guide

### 1. RoundRobinGroupChat: Sequential Collaboration

The simplest yet most predictable pattern. Agents take turns in a fixed order, ensuring everyone contributes.

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def round_robin_example():
    """Demonstrates the reflection pattern with round-robin coordination."""
    
    # Initialize model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create specialized agents
    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        system_message="""You are a creative writer. 
        Write engaging content based on the given topic.
        Focus on clarity and impact."""
    )
    
    editor = AssistantAgent(
        name="editor",
        model_client=model_client,
        system_message="""You are a meticulous editor.
        Review content for grammar, style, and clarity.
        Provide specific suggestions for improvement.
        Say 'APPROVED' when the content meets high standards."""
    )
    
    fact_checker = AssistantAgent(
        name="fact_checker",
        model_client=model_client,
        system_message="""You are a fact-checker.
        Verify claims and ensure accuracy.
        Flag any statements that need verification or correction."""
    )
    
    # Define termination conditions
    approval_termination = TextMentionTermination("APPROVED")
    max_rounds = MaxMessageTermination(max_messages=12)
    termination = approval_termination | max_rounds
    
    # Create the team
    team = RoundRobinGroupChat(
        participants=[writer, editor, fact_checker],
        termination_condition=termination
    )
    
    # Run the team
    result = await team.run(
        task="Write a brief article about the impact of AI on software development"
    )
    
    # The team will cycle through: writer -> editor -> fact_checker -> writer...
    # Until "APPROVED" is mentioned or max messages reached
    
    return result

# Usage
# asyncio.run(round_robin_example())
```

**Key Insights:**
- **Predictable flow**: Perfect for peer review, quality assurance workflows
- **Equal participation**: Ensures all perspectives are heard
- **Simple to debug**: Linear execution makes it easy to trace issues

### 2. SelectorGroupChat: Dynamic Intelligence

The most flexible pattern, using AI to determine who should speak next based on context.

```python
async def selector_group_chat_example():
    """Demonstrates dynamic speaker selection for complex tasks."""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create specialized agents with clear descriptions
    # These descriptions are crucial - the selector uses them!
    
    architect = AssistantAgent(
        name="architect",
        description="System architect - designs solutions and breaks down requirements",
        model_client=model_client,
        system_message="""You are a system architect.
        Break down complex requirements into clear components.
        Define interfaces and data flow.
        Create implementation roadmaps."""
    )
    
    backend_dev = AssistantAgent(
        name="backend_dev",
        description="Backend developer - implements server-side logic and APIs",
        model_client=model_client,
        system_message="""You are a backend developer.
        Implement robust server-side solutions.
        Design efficient database schemas.
        Create secure, scalable APIs."""
    )
    
    frontend_dev = AssistantAgent(
        name="frontend_dev",
        description="Frontend developer - builds user interfaces and experiences",
        model_client=model_client,
        system_message="""You are a frontend developer.
        Create intuitive, responsive user interfaces.
        Implement smooth user interactions.
        Ensure accessibility and performance."""
    )
    
    tester = AssistantAgent(
        name="tester",
        description="QA engineer - validates implementation and finds issues",
        model_client=model_client,
        system_message="""You are a QA engineer.
        Design comprehensive test strategies.
        Identify edge cases and potential failures.
        Verify requirements are met.
        Say 'SHIP IT' when quality standards are met."""
    )
    
    # Custom selector prompt for better control
    selector_prompt = """Select the most appropriate agent for the next step.

    Available agents and their roles:
    {roles}

    Current conversation:
    {history}

    Consider:
    1. What needs to be done next?
    2. Which agent's expertise best matches?
    3. Have all aspects been addressed?

    Select one agent from {participants}.
    """
    
    # Advanced: Custom selection function for specific logic
    def custom_selector(messages):
        """Override selection for specific scenarios."""
        last_message = messages[-1]
        
        # Always have architect respond first to new tasks
        if last_message.source == "user":
            return architect.name
        
        # After backend implementation, always get frontend perspective
        if last_message.source == backend_dev.name and "API" in last_message.content:
            return frontend_dev.name
        
        # Let AI decide for other cases
        return None
    
    # Create team with dynamic selection
    team = SelectorGroupChat(
        participants=[architect, backend_dev, frontend_dev, tester],
        model_client=model_client,
        selector_prompt=selector_prompt,
        selector_func=custom_selector,  # Optional custom logic
        allow_repeated_speaker=True,  # Agent can speak multiple times in a row
        termination_condition=TextMentionTermination("SHIP IT")
    )
    
    # Run complex task
    result = await team.run(
        task="Design and implement a real-time chat application with user authentication"
    )
    
    return result
```

**Advanced Selector Patterns:**

```python
# Pattern 1: Candidate filtering - narrow down selection pool
def candidate_filter(messages):
    """Dynamically determine which agents are eligible to speak."""
    last_speaker = messages[-1].source
    
    # After architect, only developers can speak
    if last_speaker == "architect":
        return ["backend_dev", "frontend_dev"]
    
    # After any developer, tester must review
    if last_speaker in ["backend_dev", "frontend_dev"]:
        return ["tester", "architect"]  # Tester or back to architect
    
    # Default: all agents eligible
    return None

# Pattern 2: State-based selection
class StatefulSelector:
    def __init__(self):
        self.phase = "design"  # design -> implement -> test -> deploy
        
    def __call__(self, messages):
        last_msg = messages[-1].content.lower()
        
        # Transition phases based on keywords
        if "implementation complete" in last_msg:
            self.phase = "test"
            return "tester"
        elif "tests passing" in last_msg:
            self.phase = "deploy"
            return "devops"
        
        # Phase-specific selection
        if self.phase == "design":
            return "architect" if len(messages) % 2 == 0 else "product_manager"
        
        return None
```

### 3. SwarmGroupChat: Explicit Handoffs

Perfect for workflows where agents need to explicitly pass control, maintaining clear responsibility.

```python
from autogen_agentchat.messages import HandoffMessage

async def swarm_example():
    """Demonstrates explicit handoff patterns for customer service."""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create specialized support agents
    receptionist = AssistantAgent(
        name="receptionist",
        model_client=model_client,
        system_message="""You are the first point of contact.
        Greet customers warmly and understand their needs.
        
        Route to:
        - 'technical_support' for technical issues
        - 'billing' for payment questions
        - 'sales' for new purchases
        
        Use HandoffMessage(target='agent_name', message='context') to transfer."""
    )
    
    technical_support = AssistantAgent(
        name="technical_support",
        model_client=model_client,
        system_message="""You are a technical support specialist.
        Solve technical problems step by step.
        If issue involves billing, use HandoffMessage(target='billing').
        If resolved, use HandoffMessage(target='receptionist', message='Issue resolved')."""
    )
    
    billing = AssistantAgent(
        name="billing",
        model_client=model_client,
        system_message="""You are a billing specialist.
        Handle payment issues, refunds, and subscription questions.
        After resolution, use HandoffMessage(target='receptionist', message='Billing resolved')."""
    )
    
    sales = AssistantAgent(
        name="sales",
        model_client=model_client,
        system_message="""You are a sales representative.
        Help customers find the right product or plan.
        After sale or if no sale, use HandoffMessage(target='receptionist', message='Sales complete')."""
    )
    
    # Swarm requires first agent to handle initial message
    team = SwarmGroupChat(
        participants=[receptionist, technical_support, billing, sales],
        termination_condition=MaxMessageTermination(max_messages=10)
    )
    
    # Customer query flows through appropriate agents
    result = await team.run(
        task="My subscription isn't working and I think I was charged twice last month"
    )
    
    # Flow: receptionist -> technical_support (for "not working") 
    #       -> billing (for "charged twice") -> receptionist
    
    return result
```

### 4. MagenticOneGroupChat: Intelligent Orchestration

The most sophisticated pattern, based on Microsoft Research's Magentic-One system.

```python
async def magentic_one_example():
    """Demonstrates intelligent orchestration with progress tracking."""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create a diverse team of specialists
    researcher = AssistantAgent(
        name="researcher",
        model_client=model_client,
        tools=[web_search_tool, arxiv_search_tool],
        system_message="You research and gather information from various sources."
    )
    
    analyst = AssistantAgent(
        name="analyst",
        model_client=model_client,
        tools=[data_analysis_tool, visualization_tool],
        system_message="You analyze data and extract insights."
    )
    
    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        system_message="You synthesize findings into clear reports."
    )
    
    critic = AssistantAgent(
        name="critic",
        model_client=model_client,
        system_message="You review work critically and identify gaps."
    )
    
    # MagenticOne configuration
    team = MagenticOneGroupChat(
        participants=[researcher, analyst, writer, critic],
        model_client=model_client,
        max_stalls=3,  # Replanning attempts if progress stalls
        final_answer_prompt="""Based on all the work done, provide a comprehensive 
        final answer that addresses the original request completely."""
    )
    
    # Complex, open-ended task
    result = await team.run(
        task="""Analyze the current state of quantum computing, its potential impact 
        on cryptography, and recommend preparation strategies for organizations."""
    )
    
    # The orchestrator will:
    # 1. Track progress through a ledger
    # 2. Detect when agents are stuck
    # 3. Replan if necessary
    # 4. Ensure comprehensive coverage
    
    return result
```

### 5. GraphFlow: Complex Workflows (Experimental)

For sophisticated control flow with branching, loops, and parallel execution.

```python
from autogen_agentchat.teams import GraphFlowManager
import networkx as nx

async def graph_flow_example():
    """Demonstrates graph-based workflow with complex control flow."""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create agents
    agents = {
        "validator": AssistantAgent("validator", model_client=model_client),
        "preprocessor": AssistantAgent("preprocessor", model_client=model_client),
        "analyzer_a": AssistantAgent("analyzer_a", model_client=model_client),
        "analyzer_b": AssistantAgent("analyzer_b", model_client=model_client),
        "synthesizer": AssistantAgent("synthesizer", model_client=model_client),
        "reviewer": AssistantAgent("reviewer", model_client=model_client),
    }
    
    # Build workflow graph
    workflow = nx.DiGraph()
    
    # Add nodes (agents)
    for name, agent in agents.items():
        workflow.add_node(name, agent=agent)
    
    # Define edges with conditions
    workflow.add_edge("validator", "preprocessor", 
                     condition=lambda msg: "valid" in msg.content.lower())
    workflow.add_edge("validator", "END", 
                     condition=lambda msg: "invalid" in msg.content.lower())
    
    # Parallel analysis branches
    workflow.add_edge("preprocessor", "analyzer_a")
    workflow.add_edge("preprocessor", "analyzer_b")
    
    # Convergence point
    workflow.add_edge("analyzer_a", "synthesizer")
    workflow.add_edge("analyzer_b", "synthesizer")
    
    # Review loop
    workflow.add_edge("synthesizer", "reviewer")
    workflow.add_edge("reviewer", "synthesizer",
                     condition=lambda msg: "revise" in msg.content.lower())
    workflow.add_edge("reviewer", "END",
                     condition=lambda msg: "approved" in msg.content.lower())
    
    # Create graph-based team
    team = GraphFlowManager(
        graph=workflow,
        model_client=model_client
    )
    
    result = await team.run(task="Process and analyze this dataset...")
    
    return result
```

---

## Advanced Patterns

### Nested Teams: Hierarchical Organization

Teams can contain other teams, enabling sophisticated organizational structures.

```python
async def nested_teams_example():
    """Demonstrates hierarchical team organization."""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create Research Team
    research_lead = AssistantAgent("research_lead", model_client=model_client)
    researcher_1 = AssistantAgent("researcher_1", model_client=model_client)
    researcher_2 = AssistantAgent("researcher_2", model_client=model_client)
    
    research_team = RoundRobinGroupChat(
        participants=[research_lead, researcher_1, researcher_2],
        termination_condition=TextMentionTermination("RESEARCH_COMPLETE")
    )
    
    # Create Development Team
    dev_lead = AssistantAgent("dev_lead", model_client=model_client)
    developer_1 = AssistantAgent("developer_1", model_client=model_client)
    developer_2 = AssistantAgent("developer_2", model_client=model_client)
    
    dev_team = SelectorGroupChat(
        participants=[dev_lead, developer_1, developer_2],
        model_client=model_client,
        termination_condition=TextMentionTermination("DEV_COMPLETE")
    )
    
    # Create Executive Team with nested teams
    cto = AssistantAgent(
        "cto",
        model_client=model_client,
        system_message="You coordinate between research and development teams."
    )
    
    # Teams can be participants!
    executive_team = SelectorGroupChat(
        participants=[cto, research_team, dev_team],
        model_client=model_client,
        termination_condition=MaxMessageTermination(max_messages=20)
    )
    
    result = await executive_team.run(
        task="Develop a new AI-powered feature for our product"
    )
    
    return result
```

### Custom Termination Conditions

Create sophisticated stopping criteria beyond simple text matching.

```python
from autogen_agentchat.conditions import TerminationCondition
from autogen_agentchat.messages import ChatMessage

class ConsensusTermination(TerminationCondition):
    """Terminate when all agents agree."""
    
    def __init__(self, required_agreements: int = 3):
        self.required_agreements = required_agreements
        self.agreements = set()
    
    async def is_terminal(self, messages: list[ChatMessage]) -> bool:
        if not messages:
            return False
        
        last_message = messages[-1]
        
        # Check for agreement signals
        if "i agree" in last_message.content.lower():
            self.agreements.add(last_message.source)
        elif "i disagree" in last_message.content.lower():
            self.agreements.discard(last_message.source)
        
        # Terminal when enough agents agree
        return len(self.agreements) >= self.required_agreements
    
    def reset(self):
        self.agreements.clear()


class QualityGateTermination(TerminationCondition):
    """Terminate when quality metrics are met."""
    
    def __init__(self, min_score: float = 0.8):
        self.min_score = min_score
    
    async def is_terminal(self, messages: list[ChatMessage]) -> bool:
        # Look for quality scores in messages
        for msg in reversed(messages):
            if "quality_score:" in msg.content:
                score = float(msg.content.split("quality_score:")[1].split()[0])
                return score >= self.min_score
        return False


# Combine conditions
consensus = ConsensusTermination(required_agreements=3)
quality = QualityGateTermination(min_score=0.9)
timeout = MaxMessageTermination(max_messages=50)

# Must meet consensus AND quality, OR timeout
termination = (consensus & quality) | timeout
```

### State Management Across Teams

Maintain context and state between team executions.

```python
class StatefulTeamCoordinator:
    """Manages state across multiple team executions."""
    
    def __init__(self):
        self.global_context = {}
        self.execution_history = []
        
    async def run_pipeline(self, tasks: list[dict]):
        """Run multiple teams in sequence with shared state."""
        
        for task_config in tasks:
            team = task_config["team"]
            task = task_config["task"]
            
            # Inject context into task
            enriched_task = f"{task}\n\nContext: {json.dumps(self.global_context)}"
            
            # Run team
            result = await team.run(task=enriched_task)
            
            # Extract and store results
            self.execution_history.append(result)
            self._update_context(result)
            
            # Check if we should continue
            if not self._should_continue(result):
                break
        
        return self.execution_history
    
    def _update_context(self, result: TaskResult):
        """Extract key information from results."""
        for message in result.messages:
            # Extract structured data
            if "```json" in message.content:
                json_str = message.content.split("```json")[1].split("```")[0]
                data = json.loads(json_str)
                self.global_context.update(data)
    
    def _should_continue(self, result: TaskResult) -> bool:
        """Determine if pipeline should continue."""
        return "STOP_PIPELINE" not in result.messages[-1].content
```

### Performance Monitoring

Track and optimize team performance.

```python
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class TeamMetrics:
    """Performance metrics for team execution."""
    total_messages: int
    execution_time: float
    tokens_used: Dict[str, int]
    agent_participation: Dict[str, int]
    termination_reason: str

class MonitoredTeam:
    """Wrapper for teams with performance monitoring."""
    
    def __init__(self, team):
        self.team = team
        self.metrics_history: List[TeamMetrics] = []
    
    async def run_with_metrics(self, task: str) -> tuple[TaskResult, TeamMetrics]:
        """Run team and collect metrics."""
        
        start_time = time.time()
        agent_participation = {}
        tokens_used = {}
        
        # Run team with streaming to collect metrics
        messages = []
        async for msg in self.team.run_stream(task=task):
            if isinstance(msg, TaskResult):
                result = msg
                break
            
            messages.append(msg)
            
            # Track participation
            if hasattr(msg, 'source'):
                agent_participation[msg.source] = agent_participation.get(msg.source, 0) + 1
            
            # Track token usage
            if hasattr(msg, 'models_usage') and msg.models_usage:
                tokens_used[msg.source] = tokens_used.get(msg.source, 0) + \
                    msg.models_usage.prompt_tokens + msg.models_usage.completion_tokens
        
        # Calculate metrics
        metrics = TeamMetrics(
            total_messages=len(messages),
            execution_time=time.time() - start_time,
            tokens_used=tokens_used,
            agent_participation=agent_participation,
            termination_reason=result.stop_reason
        )
        
        self.metrics_history.append(metrics)
        
        return result, metrics
    
    def analyze_performance(self) -> dict:
        """Analyze team performance over multiple runs."""
        
        if not self.metrics_history:
            return {}
        
        return {
            "avg_messages": sum(m.total_messages for m in self.metrics_history) / len(self.metrics_history),
            "avg_execution_time": sum(m.execution_time for m in self.metrics_history) / len(self.metrics_history),
            "total_tokens": sum(sum(m.tokens_used.values()) for m in self.metrics_history),
            "most_active_agent": self._get_most_active_agent(),
            "common_termination": self._get_common_termination()
        }
    
    def _get_most_active_agent(self) -> str:
        """Identify the most active agent across all runs."""
        total_participation = {}
        for metrics in self.metrics_history:
            for agent, count in metrics.agent_participation.items():
                total_participation[agent] = total_participation.get(agent, 0) + count
        return max(total_participation, key=total_participation.get) if total_participation else "N/A"
    
    def _get_common_termination(self) -> str:
        """Find most common termination reason."""
        reasons = [m.termination_reason for m in self.metrics_history]
        return max(set(reasons), key=reasons.count) if reasons else "N/A"
```

---

## Practical Examples

### Example 1: Research Team

A complete research team that gathers and synthesizes information.

```python
async def research_team_example():
    """Multi-agent research team with web search and synthesis."""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Define research tools
    def web_search(query: str) -> str:
        """Simulated web search tool."""
        # In production, use real search API
        return f"Search results for '{query}': [simulated results]"
    
    def arxiv_search(query: str) -> str:
        """Search academic papers."""
        return f"Academic papers about '{query}': [simulated papers]"
    
    def summarize_text(text: str, max_words: int = 100) -> str:
        """Summarize long text."""
        return f"Summary (max {max_words} words): {text[:max_words]}..."
    
    # Create specialized researchers
    web_researcher = AssistantAgent(
        name="web_researcher",
        description="Searches and analyzes web content",
        model_client=model_client,
        tools=[web_search],
        system_message="""You are a web research specialist.
        Search for current information and trends.
        Focus on authoritative sources.
        Provide factual, well-sourced information."""
    )
    
    academic_researcher = AssistantAgent(
        name="academic_researcher",
        description="Searches and analyzes academic literature",
        model_client=model_client,
        tools=[arxiv_search],
        system_message="""You are an academic research specialist.
        Find and analyze peer-reviewed papers.
        Focus on methodology and evidence.
        Cite sources properly."""
    )
    
    synthesizer = AssistantAgent(
        name="synthesizer",
        description="Combines findings into coherent insights",
        model_client=model_client,
        tools=[summarize_text],
        system_message="""You are a research synthesizer.
        Combine findings from multiple sources.
        Identify patterns and contradictions.
        Create comprehensive summaries.
        Say 'RESEARCH COMPLETE' when synthesis is final."""
    )
    
    critic = AssistantAgent(
        name="critic",
        description="Reviews research for gaps and bias",
        model_client=model_client,
        system_message="""You are a research critic.
        Identify gaps in the research.
        Check for bias and conflicting information.
        Suggest additional areas to explore.
        Ensure comprehensive coverage."""
    )
    
    # Dynamic selection based on research needs
    selector_prompt = """Select the next researcher based on what's needed:
    
    {roles}
    
    Current research progress:
    {history}
    
    Consider:
    - What information is still missing?
    - What type of source would be most valuable?
    - Has the research been critically reviewed?
    
    Select from {participants}."""
    
    # Create research team
    research_team = SelectorGroupChat(
        participants=[web_researcher, academic_researcher, synthesizer, critic],
        model_client=model_client,
        selector_prompt=selector_prompt,
        termination_condition=TextMentionTermination("RESEARCH COMPLETE"),
        allow_repeated_speaker=True
    )
    
    # Run research task
    result = await research_team.run(
        task="Research the impact of large language models on software development practices"
    )
    
    return result
```

### Example 2: Software Development Team

A complete development team with planning, implementation, and testing.

```python
async def development_team_example():
    """Full software development team with code generation and testing."""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Development tools
    def create_file(path: str, content: str) -> str:
        """Create a file with content."""
        # Simulated file creation
        return f"Created {path}"
    
    def run_tests(test_file: str) -> dict:
        """Run tests and return results."""
        # Simulated test execution
        return {"passed": 8, "failed": 2, "coverage": 85}
    
    def lint_code(file_path: str) -> list:
        """Check code quality."""
        # Simulated linting
        return ["Line 10: Missing docstring", "Line 25: Variable unused"]
    
    # Create development team
    product_manager = AssistantAgent(
        name="product_manager",
        description="Defines requirements and acceptance criteria",
        model_client=model_client,
        system_message="""You are a product manager.
        Define clear requirements and user stories.
        Create acceptance criteria.
        Prioritize features."""
    )
    
    architect = AssistantAgent(
        name="architect",
        description="Designs system architecture and interfaces",
        model_client=model_client,
        system_message="""You are a software architect.
        Design scalable system architectures.
        Define APIs and data models.
        Consider performance and security."""
    )
    
    backend_developer = AssistantAgent(
        name="backend_developer",
        description="Implements server-side logic",
        model_client=model_client,
        tools=[create_file],
        system_message="""You are a backend developer.
        Write clean, efficient Python code.
        Implement robust error handling.
        Follow REST API best practices."""
    )
    
    frontend_developer = AssistantAgent(
        name="frontend_developer",
        description="Implements user interfaces",
        model_client=model_client,
        tools=[create_file],
        system_message="""You are a frontend developer.
        Create responsive React components.
        Implement smooth user interactions.
        Ensure accessibility."""
    )
    
    qa_engineer = AssistantAgent(
        name="qa_engineer",
        description="Tests and validates implementation",
        model_client=model_client,
        tools=[run_tests, lint_code],
        system_message="""You are a QA engineer.
        Write comprehensive test cases.
        Perform integration testing.
        Check code quality.
        Say 'READY TO DEPLOY' when all tests pass."""
    )
    
    # Create team with custom selection logic
    def dev_team_selector(messages):
        """Smart selection for development workflow."""
        if not messages:
            return product_manager.name
        
        last = messages[-1]
        
        # Requirements -> Architecture
        if last.source == product_manager.name and "requirements" in last.content.lower():
            return architect.name
        
        # Architecture -> Implementation
        if last.source == architect.name and "design" in last.content.lower():
            # Parallel development possible
            import random
            return random.choice([backend_developer.name, frontend_developer.name])
        
        # After implementation -> Testing
        if last.source in [backend_developer.name, frontend_developer.name]:
            # Check if both have implemented
            sources = [m.source for m in messages[-5:]]
            if backend_developer.name in sources and frontend_developer.name in sources:
                return qa_engineer.name
        
        return None  # Let AI decide
    
    # Create development team
    dev_team = SelectorGroupChat(
        participants=[
            product_manager,
            architect,
            backend_developer,
            frontend_developer,
            qa_engineer
        ],
        model_client=model_client,
        selector_func=dev_team_selector,
        termination_condition=TextMentionTermination("READY TO DEPLOY"),
        allow_repeated_speaker=True
    )
    
    # Run development task
    result = await dev_team.run(
        task="Build a REST API for a todo list application with user authentication"
    )
    
    return result
```

### Example 3: Creative Writing Team

A creative team for content generation with multiple review stages.

```python
async def creative_writing_team():
    """Creative writing team with ideation, drafting, and editing."""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Creative tools
    def word_count(text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def sentiment_analysis(text: str) -> dict:
        """Analyze emotional tone."""
        # Simulated sentiment analysis
        return {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
    
    brainstormer = AssistantAgent(
        name="brainstormer",
        description="Generates creative ideas and concepts",
        model_client=model_client,
        system_message="""You are a creative brainstormer.
        Generate unique, engaging ideas.
        Think outside the box.
        Consider multiple angles and perspectives."""
    )
    
    storyteller = AssistantAgent(
        name="storyteller",
        description="Crafts compelling narratives",
        model_client=model_client,
        tools=[word_count],
        system_message="""You are a master storyteller.
        Create engaging narratives with strong characters.
        Build tension and emotional resonance.
        Use vivid, sensory language."""
    )
    
    editor = AssistantAgent(
        name="editor",
        description="Refines and polishes content",
        model_client=model_client,
        tools=[word_count, sentiment_analysis],
        system_message="""You are a meticulous editor.
        Improve clarity and flow.
        Fix grammar and style issues.
        Ensure consistent tone and voice.
        Say 'PUBLICATION READY' when content meets standards."""
    )
    
    sensitivity_reader = AssistantAgent(
        name="sensitivity_reader",
        description="Reviews content for inclusivity",
        model_client=model_client,
        system_message="""You are a sensitivity reader.
        Check for inclusive language.
        Identify potential biases.
        Suggest respectful alternatives.
        Ensure diverse representation."""
    )
    
    # Round-robin for equal creative input
    creative_team = RoundRobinGroupChat(
        participants=[brainstormer, storyteller, editor, sensitivity_reader],
        termination_condition=TextMentionTermination("PUBLICATION READY") | 
                             MaxMessageTermination(max_messages=16)
    )
    
    result = await creative_team.run(
        task="Write a short story about an AI learning to understand human emotions"
    )
    
    return result
```

---

## Optimization & Debugging

### Performance Optimization

```python
class OptimizedTeamConfig:
    """Configuration patterns for optimal team performance."""
    
    @staticmethod
    def minimize_token_usage(team):
        """Configure team for minimal token consumption."""
        
        # Use concise system messages
        for agent in team.participants:
            if hasattr(agent, 'system_message'):
                agent.system_message = agent.system_message[:200]  # Truncate
        
        # Add token-aware termination
        class TokenLimitTermination(TerminationCondition):
            def __init__(self, max_tokens: int = 10000):
                self.total_tokens = 0
                self.max_tokens = max_tokens
            
            async def is_terminal(self, messages):
                for msg in messages:
                    if hasattr(msg, 'models_usage') and msg.models_usage:
                        self.total_tokens += msg.models_usage.prompt_tokens
                        self.total_tokens += msg.models_usage.completion_tokens
                
                return self.total_tokens >= self.max_tokens
        
        # Add token limit to termination conditions
        team.termination_condition = team.termination_condition | TokenLimitTermination()
        
        return team
    
    @staticmethod
    def maximize_parallelism(agents: list) -> SelectorGroupChat:
        """Configure for parallel execution where possible."""
        
        def parallel_selector(messages):
            """Select multiple agents for parallel execution."""
            # This is conceptual - actual implementation would need
            # parallel execution support in the framework
            
            last_msg = messages[-1] if messages else None
            if not last_msg:
                return agents[0].name
            
            # Identify independent tasks
            if "analyze" in last_msg.content.lower():
                # Multiple analysts can work in parallel
                return [a.name for a in agents if "analyst" in a.name]
            
            return None
        
        return SelectorGroupChat(
            participants=agents,
            selector_func=parallel_selector,
            allow_repeated_speaker=False
        )
```

### Debugging Team Interactions

```python
class TeamDebugger:
    """Utilities for debugging team behavior."""
    
    @staticmethod
    async def trace_execution(team, task: str):
        """Detailed execution trace for debugging."""
        
        print(f"\n{'='*50}")
        print(f"TASK: {task}")
        print(f"TEAM: {team.__class__.__name__}")
        print(f"PARTICIPANTS: {[p.name for p in team.participants]}")
        print(f"{'='*50}\n")
        
        message_count = 0
        speaker_sequence = []
        
        async for msg in team.run_stream(task=task):
            if isinstance(msg, TaskResult):
                print(f"\n{'='*50}")
                print(f"EXECUTION COMPLETE")
                print(f"Stop Reason: {msg.stop_reason}")
                print(f"Total Messages: {message_count}")
                print(f"Speaker Sequence: {' -> '.join(speaker_sequence)}")
                print(f"{'='*50}")
                return msg
            
            message_count += 1
            
            # Print message details
            print(f"\nMessage #{message_count}")
            print(f"Type: {type(msg).__name__}")
            
            if hasattr(msg, 'source'):
                print(f"Source: {msg.source}")
                speaker_sequence.append(msg.source)
            
            if hasattr(msg, 'content'):
                print(f"Content Preview: {msg.content[:200]}...")
            
            if hasattr(msg, 'models_usage') and msg.models_usage:
                print(f"Tokens: {msg.models_usage.prompt_tokens} prompt, "
                      f"{msg.models_usage.completion_tokens} completion")
    
    @staticmethod
    def analyze_team_dynamics(result: TaskResult):
        """Analyze team interaction patterns."""
        
        analysis = {
            "total_messages": len(result.messages),
            "unique_speakers": set(),
            "speaker_frequency": {},
            "message_types": {},
            "interaction_graph": []
        }
        
        prev_speaker = None
        
        for msg in result.messages:
            # Track speakers
            if hasattr(msg, 'source'):
                analysis["unique_speakers"].add(msg.source)
                analysis["speaker_frequency"][msg.source] = \
                    analysis["speaker_frequency"].get(msg.source, 0) + 1
                
                # Track interactions
                if prev_speaker and prev_speaker != msg.source:
                    analysis["interaction_graph"].append(
                        (prev_speaker, msg.source)
                    )
                prev_speaker = msg.source
            
            # Track message types
            msg_type = type(msg).__name__
            analysis["message_types"][msg_type] = \
                analysis["message_types"].get(msg_type, 0) + 1
        
        return analysis
```

### Testing Team Configurations

```python
import pytest
from unittest.mock import Mock, AsyncMock

async def test_team_configuration():
    """Test team setup and configuration."""
    
    # Create mock model client
    mock_model = Mock()
    mock_model.create = AsyncMock(return_value=Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    ))
    
    # Create test agents
    agent1 = AssistantAgent("agent1", model_client=mock_model)
    agent2 = AssistantAgent("agent2", model_client=mock_model)
    
    # Test team creation
    team = RoundRobinGroupChat(
        participants=[agent1, agent2],
        termination_condition=MaxMessageTermination(max_messages=4)
    )
    
    # Run test task
    result = await team.run(task="Test task")
    
    # Assertions
    assert len(result.messages) <= 4
    assert result.stop_reason is not None
    
    # Test reset
    await team.reset()
    assert len(team._message_thread) == 0

async def test_selector_logic():
    """Test custom selector function."""
    
    def test_selector(messages):
        if not messages:
            return "agent1"
        return "agent2" if messages[-1].source == "agent1" else "agent1"
    
    mock_model = Mock()
    agent1 = AssistantAgent("agent1", model_client=mock_model)
    agent2 = AssistantAgent("agent2", model_client=mock_model)
    
    team = SelectorGroupChat(
        participants=[agent1, agent2],
        model_client=mock_model,
        selector_func=test_selector,
        termination_condition=MaxMessageTermination(max_messages=4)
    )
    
    # Verify selector is called correctly
    assert test_selector([]) == "agent1"
    assert test_selector([Mock(source="agent1")]) == "agent2"
```

### Common Issues and Solutions

```python
class TeamTroubleshooting:
    """Common team issues and their solutions."""
    
    @staticmethod
    def fix_infinite_loops(team):
        """Prevent teams from getting stuck in loops."""
        
        # Add multiple termination conditions
        team.termination_condition = (
            team.termination_condition |
            MaxMessageTermination(max_messages=50) |
            ExternalTermination()  # Allow manual stopping
        )
        
        # Add loop detection
        class LoopDetectionTermination(TerminationCondition):
            def __init__(self, pattern_length: int = 3, max_repeats: int = 2):
                self.pattern_length = pattern_length
                self.max_repeats = max_repeats
            
            async def is_terminal(self, messages):
                if len(messages) < self.pattern_length * self.max_repeats:
                    return False
                
                # Check for repeating patterns
                recent = messages[-(self.pattern_length * self.max_repeats):]
                patterns = []
                
                for i in range(0, len(recent), self.pattern_length):
                    pattern = tuple(m.source for m in recent[i:i+self.pattern_length]
                                  if hasattr(m, 'source'))
                    patterns.append(pattern)
                
                # Check if pattern repeats
                return len(set(patterns)) == 1 and len(patterns) == self.max_repeats
        
        team.termination_condition = team.termination_condition | LoopDetectionTermination()
        
        return team
    
    @staticmethod
    def improve_context_awareness(team):
        """Enhance agents' awareness of conversation context."""
        
        # Add context summarizer
        context_agent = AssistantAgent(
            name="context_manager",
            model_client=team.model_client,
            system_message="""You maintain conversation context.
            Every 5 messages, provide a brief summary of:
            1. What has been accomplished
            2. What remains to be done
            3. Key decisions made
            Keep summaries under 100 words."""
        )
        
        # Inject context agent periodically
        original_selector = team.selector_func if hasattr(team, 'selector_func') else None
        
        def context_aware_selector(messages):
            # Inject context manager every 5 messages
            if len(messages) % 5 == 0 and len(messages) > 0:
                return context_agent.name
            
            # Otherwise use original selection
            return original_selector(messages) if original_selector else None
        
        if hasattr(team, 'selector_func'):
            team.selector_func = context_aware_selector
        
        return team
```

---

## Best Practices Summary

### Do's ✅

1. **Start Simple**: Begin with RoundRobinGroupChat, evolve to SelectorGroupChat as needed
2. **Clear Agent Descriptions**: Essential for selector-based teams
3. **Meaningful Termination**: Define clear success criteria
4. **Reset Between Tasks**: Unless tasks are related
5. **Monitor Performance**: Track tokens, time, and participation
6. **Test Configurations**: Verify team behavior before production

### Don'ts ❌

1. **Over-engineer**: Don't use teams for simple single-agent tasks
2. **Ignore Context Limits**: Monitor message history size
3. **Forget Error Handling**: Teams can fail - plan for it
4. **Skip Testing**: Test selector logic and termination conditions
5. **Neglect Performance**: Teams can be expensive - optimize token usage

### When to Use Each Team Type

| Scenario | Recommended Team | Why |
|----------|------------------|-----|
| Code review | RoundRobinGroupChat | Predictable, everyone reviews |
| Customer support | SwarmGroupChat | Clear handoff paths |
| Research | SelectorGroupChat | Dynamic expertise needed |
| Complex problem-solving | MagenticOneGroupChat | Needs intelligent orchestration |
| CI/CD pipeline | GraphFlow | Complex branching logic |
| Brainstorming | RoundRobinGroupChat | Equal participation |
| Debugging | SelectorGroupChat | Context-dependent expertise |

---

## Conclusion

AutoGen Teams provide powerful patterns for multi-agent coordination. Start with simple patterns like RoundRobinGroupChat, experiment with dynamic selection via SelectorGroupChat, and evolve to sophisticated orchestration with MagenticOneGroupChat as your needs grow.

Remember: The best team configuration depends on your specific use case. Test different patterns, measure performance, and iterate based on results.

Happy team building! 🚀