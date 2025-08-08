# CLAUDE.md - AutoGen Configuration Patterns Guide

*As your AutoGen guide, I'll help you master configuration patterns - from simple setups to complex multi-provider deployments. Good configuration is the foundation of reliable AutoGen applications.*

This guide covers configuration patterns and best practices for AutoGen applications.

## Overview

AutoGen supports multiple configuration approaches: direct Python code, YAML files, environment variables, and component serialization. Choosing the right pattern depends on your deployment needs, security requirements, and team workflows.

## Configuration Approaches

### 1. Direct Python Configuration

Best for: Development, simple applications, quick prototypes

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent

# Direct configuration
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key="sk-...",  # Don't hardcode in production!
    temperature=0.7,
    max_tokens=2000,
    timeout=30,
    max_retries=3
)

agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a helpful assistant."
)
```

### 2. Environment Variables

Best for: Production, security, CI/CD

```python
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Use environment variables
model_client = OpenAIChatCompletionClient(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY"),  # Required
    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
)
```

**.env file approach**:
```bash
# .env file
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com
AZURE_OPENAI_API_KEY=...
```

```python
from dotenv import load_dotenv
load_dotenv()  # Load .env file

# Now environment variables are available
model_client = OpenAIChatCompletionClient(
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### 3. YAML Configuration

Best for: Complex configurations, multi-environment deployments

**model_config.yaml**:
```yaml
# Model client configuration
provider: autogen_ext.models.openai.OpenAIChatCompletionClient
config:
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}  # Environment variable reference
  temperature: 0.7
  max_tokens: 2000
  timeout: 30
  max_retries: 3
  
# Optional: Response format
response_format:
  type: json_object
```

**Loading YAML config**:
```python
from autogen_agentchat.models import ChatCompletionClient
import yaml
import os

# Load and parse YAML
with open("model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Replace environment variables
def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var)
    return config

config = replace_env_vars(config)

# Create model client
model_client = ChatCompletionClient.load_component(config)
```

### 4. Component Serialization

Best for: Saving/loading agent configurations, reproducible setups

```python
from autogen_core import Component

# Save configuration
agent_config = agent.dump_component()

# Save to file
import json
with open("agent_config.json", "w") as f:
    json.dump(agent_config, f, indent=2)

# Load from file
with open("agent_config.json", "r") as f:
    loaded_config = json.load(f)

# Recreate agent
loaded_agent = Component.load_component(loaded_config)
```

## Multi-Provider Configuration

### Pattern 1: Provider Registry

```python
from typing import Dict, Any
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.azure import AzureOpenAIChatCompletionClient

class ModelRegistry:
    """Centralized model client management."""
    
    def __init__(self, config_path: str = "models.yaml"):
        self.config = self._load_config(config_path)
        self._clients = {}
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    
    def get_client(self, name: str):
        """Get or create a model client by name."""
        if name not in self._clients:
            client_config = self.config["models"][name]
            provider = client_config["provider"]
            
            if provider == "openai":
                self._clients[name] = OpenAIChatCompletionClient(**client_config["config"])
            elif provider == "anthropic":
                self._clients[name] = AnthropicChatCompletionClient(**client_config["config"])
            elif provider == "azure":
                self._clients[name] = AzureOpenAIChatCompletionClient(**client_config["config"])
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        return self._clients[name]

# Usage
registry = ModelRegistry()
fast_model = registry.get_client("fast")
smart_model = registry.get_client("smart")
```

**models.yaml**:
```yaml
models:
  fast:
    provider: openai
    config:
      model: gpt-3.5-turbo
      api_key: ${OPENAI_API_KEY}
      temperature: 0.0
      
  smart:
    provider: anthropic
    config:
      model: claude-3-5-sonnet-20241022
      api_key: ${ANTHROPIC_API_KEY}
      max_tokens: 4000
      
  vision:
    provider: openai
    config:
      model: gpt-4-vision-preview
      api_key: ${OPENAI_API_KEY}
```

### Pattern 2: Environment-Based Configuration

```python
import os

class EnvironmentConfig:
    """Configuration based on deployment environment."""
    
    def __init__(self):
        self.env = os.getenv("AUTOGEN_ENV", "development")
        self.config = self._load_env_config()
    
    def _load_env_config(self):
        configs = {
            "development": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.9,
                "cache_enabled": True,
                "log_level": "DEBUG"
            },
            "staging": {
                "model": "gpt-4o",
                "temperature": 0.7,
                "cache_enabled": True,
                "log_level": "INFO"
            },
            "production": {
                "model": "gpt-4-turbo",
                "temperature": 0.3,
                "cache_enabled": False,
                "log_level": "WARNING"
            }
        }
        return configs.get(self.env, configs["development"])
    
    def get_model_client(self):
        return OpenAIChatCompletionClient(
            model=self.config["model"],
            temperature=self.config["temperature"]
        )
```

## Team Configuration Patterns

### Pattern 1: Team Configuration File

**team_config.yaml**:
```yaml
team:
  name: "Research Team"
  type: "RoundRobinGroupChat"
  termination:
    type: "MaxMessageTermination"
    max_messages: 20
    
agents:
  researcher:
    type: "AssistantAgent"
    model: "smart"  # Reference to model config
    system_message: |
      You are a research specialist.
      Focus on finding accurate, relevant information.
    tools:
      - "web_search"
      - "paper_search"
      
  analyst:
    type: "AssistantAgent"
    model: "smart"
    system_message: |
      You are a data analyst.
      Analyze information and provide insights.
    tools:
      - "data_analysis"
      - "visualization"
      
  writer:
    type: "AssistantAgent"
    model: "fast"
    system_message: |
      You are a technical writer.
      Create clear, concise documentation.
```

**Loading team configuration**:
```python
class TeamFactory:
    """Create teams from configuration."""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.tool_registry = {}  # Tool definitions
    
    def create_team(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Create agents
        agents = []
        for agent_name, agent_config in config["agents"].items():
            model_client = self.model_registry.get_client(agent_config["model"])
            
            agent = AssistantAgent(
                name=agent_name,
                model_client=model_client,
                system_message=agent_config["system_message"],
                tools=self._get_tools(agent_config.get("tools", []))
            )
            agents.append(agent)
        
        # Create team
        team_config = config["team"]
        if team_config["type"] == "RoundRobinGroupChat":
            termination = self._create_termination(team_config["termination"])
            return RoundRobinGroupChat(
                participants=agents,
                termination_condition=termination
            )
```

### Pattern 2: Hierarchical Configuration

```python
# Base configuration
base_config = {
    "model_defaults": {
        "temperature": 0.7,
        "max_tokens": 2000,
        "timeout": 30
    },
    "agent_defaults": {
        "max_consecutive_replies": 10,
        "human_input_mode": "NEVER"
    }
}

# Environment overrides
env_overrides = {
    "production": {
        "model_defaults": {
            "temperature": 0.3,  # More deterministic in production
            "timeout": 60
        }
    }
}

# Merge configurations
def merge_configs(base, override):
    """Deep merge configurations."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result

final_config = merge_configs(base_config, env_overrides.get(os.getenv("ENV", "development"), {}))
```

## Security Best Practices

### 1. Secrets Management

```python
# Use secret management services
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

class SecureConfig:
    def __init__(self, vault_url: str):
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=credential)
    
    def get_api_key(self, name: str) -> str:
        """Retrieve API key from secure vault."""
        secret = self.client.get_secret(name)
        return secret.value

# Use with model client
secure_config = SecureConfig("https://myvault.vault.azure.net/")
model_client = OpenAIChatCompletionClient(
    api_key=secure_config.get_api_key("openai-api-key")
)
```

### 2. Configuration Validation

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class ModelConfig(BaseModel):
    """Validated model configuration."""
    provider: str = Field(..., pattern="^(openai|anthropic|azure)$")
    model: str
    api_key: str = Field(..., min_length=20)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, gt=0, le=128000)
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if v.startswith("sk-proj-"):  # OpenAI project keys
            return v
        elif v.startswith("sk-"):  # OpenAI keys
            return v
        elif len(v) > 50:  # Anthropic keys are longer
            return v
        raise ValueError("Invalid API key format")

# Use validated config
config_data = {"provider": "openai", "model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}
validated_config = ModelConfig(**config_data)
```

### 3. Audit Logging

```python
import logging
from datetime import datetime

class AuditedConfig:
    """Configuration with audit trail."""
    
    def __init__(self):
        self._config = {}
        self._audit_log = []
        
    def set(self, key: str, value: Any, user: str = "system"):
        """Set configuration with audit."""
        old_value = self._config.get(key)
        self._config[key] = value
        
        self._audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "user": user,
            "action": "set",
            "key": key,
            "old_value": old_value,
            "new_value": value
        })
        
        logging.info(f"Config change: {key} changed by {user}")
    
    def get_audit_trail(self):
        return self._audit_log
```

## Dynamic Configuration

### 1. Hot Reloading

```python
import watchdog.observers
import watchdog.events

class ConfigReloader(watchdog.events.FileSystemEventHandler):
    """Automatically reload configuration on file changes."""
    
    def __init__(self, config_path: str, callback):
        self.config_path = config_path
        self.callback = callback
        self.observer = watchdog.observers.Observer()
        self.observer.schedule(self, path=os.path.dirname(config_path))
        
    def on_modified(self, event):
        if event.src_path == self.config_path:
            logging.info(f"Reloading configuration: {self.config_path}")
            self.callback()
    
    def start(self):
        self.observer.start()
    
    def stop(self):
        self.observer.stop()
        self.observer.join()

# Usage
def reload_config():
    global model_client
    model_client = load_model_client("config.yaml")

reloader = ConfigReloader("config.yaml", reload_config)
reloader.start()
```

### 2. Feature Flags

```python
class FeatureConfig:
    """Feature flag configuration."""
    
    def __init__(self, flags_path: str = "features.yaml"):
        self.flags = self._load_flags(flags_path)
    
    def _load_flags(self, path: str):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    
    def is_enabled(self, feature: str, user_id: str = None) -> bool:
        """Check if feature is enabled."""
        flag = self.flags.get(feature, {})
        
        # Global enable/disable
        if not flag.get("enabled", False):
            return False
            
        # Percentage rollout
        if "percentage" in flag and user_id:
            import hashlib
            hash_val = int(hashlib.md5(f"{feature}{user_id}".encode()).hexdigest(), 16)
            return (hash_val % 100) < flag["percentage"]
            
        # User whitelist
        if "users" in flag and user_id:
            return user_id in flag["users"]
            
        return True

# Usage
features = FeatureConfig()
if features.is_enabled("new_model", user_id="user123"):
    model_client = OpenAIChatCompletionClient(model="gpt-4-turbo-preview")
else:
    model_client = OpenAIChatCompletionClient(model="gpt-4")
```

## Testing Configuration

### 1. Configuration Tests

```python
import pytest
from unittest.mock import patch

def test_model_config_loading():
    """Test configuration loading."""
    config = {
        "provider": "autogen_ext.models.openai.OpenAIChatCompletionClient",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.7
        }
    }
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = ChatCompletionClient.load_component(config)
        assert client.model == "gpt-4o"

def test_environment_config():
    """Test environment-based configuration."""
    with patch.dict(os.environ, {"AUTOGEN_ENV": "production"}):
        config = EnvironmentConfig()
        assert config.config["temperature"] == 0.3
        assert config.config["model"] == "gpt-4-turbo"
```

### 2. Configuration Validation

```python
def validate_production_config(config_path: str):
    """Validate configuration for production."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    errors = []
    
    # Check for hardcoded secrets
    if "api_key" in str(config) and not "${" in str(config):
        errors.append("Hardcoded API key detected")
    
    # Verify required fields
    required_fields = ["provider", "model", "timeout"]
    for field in required_fields:
        if field not in config.get("config", {}):
            errors.append(f"Missing required field: {field}")
    
    # Check model validity
    valid_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    if config.get("config", {}).get("model") not in valid_models:
        errors.append("Invalid model specified")
    
    return errors
```

## Configuration Templates

### 1. Minimal Configuration

```yaml
# minimal_config.yaml
provider: autogen_ext.models.openai.OpenAIChatCompletionClient
config:
  model: gpt-3.5-turbo
  api_key: ${OPENAI_API_KEY}
```

### 2. Full Production Configuration

```yaml
# production_config.yaml
provider: autogen_ext.models.openai.OpenAIChatCompletionClient
config:
  model: gpt-4-turbo
  api_key: ${OPENAI_API_KEY}
  temperature: 0.3
  max_tokens: 4000
  timeout: 60
  max_retries: 3
  
  # Advanced options
  response_format:
    type: json_object
  
  # Rate limiting
  requests_per_minute: 50
  tokens_per_minute: 10000
  
  # Logging
  log_requests: true
  log_responses: false
  
# Fallback configuration
fallback:
  provider: autogen_ext.models.anthropic.AnthropicChatCompletionClient
  config:
    model: claude-3-5-sonnet-20241022
    api_key: ${ANTHROPIC_API_KEY}
```

### 3. Multi-Agent System Configuration

```yaml
# system_config.yaml
system:
  name: "Customer Support System"
  version: "1.0.0"
  environment: ${AUTOGEN_ENV}
  
models:
  fast:
    provider: openai
    model: gpt-3.5-turbo
    temperature: 0.0
    
  smart:
    provider: openai
    model: gpt-4-turbo
    temperature: 0.7
    
agents:
  classifier:
    model: fast
    role: "Classify customer inquiries"
    
  responder:
    model: smart
    role: "Provide detailed responses"
    
  escalator:
    model: fast
    role: "Identify issues needing human help"
    
teams:
  support:
    type: SelectorGroupChat
    agents: [classifier, responder, escalator]
    selector_model: fast
    max_messages: 20
```

Remember: Good configuration management is crucial for maintainable AutoGen applications. Start simple and add complexity as needed!