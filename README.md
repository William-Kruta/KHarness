# KHarness

A lightweight Python framework for building LLM agents with tool use, conversation memory, and multi-round research pipelines. Supports local inference via [llama.cpp](https://github.com/ggerganov/llama.cpp) and [Ollama](https://ollama.com).

---

## Installation

```bash
pip install kharness
```

**Optional dependencies** (install what you need):

```bash
pip install langchain-core          # Required for tool decorators
pip install ddgs beautifulsoup4     # Web tools
pip install yahoors                 # Stock tools
```

---

## Providers

Providers wrap a local inference backend and expose a unified `chat()` interface.

### Ollama

```python
from kharness.providers import Ollama

provider = Ollama()                         # defaults: localhost:11434
provider.set_default_model("qwen2.5:7b")

response = provider.chat(
    [{"role": "user", "content": "Hello!"}],
    model="qwen2.5:7b",
)
```

### LlamaCpp

```python
from kharness.providers import LlamaCpp

provider = LlamaCpp()                       # defaults: localhost:8080
provider.set_model_dir("/path/to/models")
provider.set_default_model("Mistral-7B-Q4_K_M.gguf")

# Start the llama-server subprocess
process = provider.start_llama_server()

# Check if the server is ready
if provider.check_health():
    response = provider.chat(
        [{"role": "user", "content": "Hello!"}]
    )
```

**Constructor options:**

| Parameter           | Default              | Description                             |
| ------------------- | -------------------- | --------------------------------------- |
| `server_url`        | `"http://localhost"` | Server host                             |
| `port`              | `"8080"`             | Server port                             |
| `max_iterations`    | `5`                  | Max tool-call loop iterations           |
| `strip_tools_after` | `3`                  | Force text response after N tool rounds |

---

## Agent

`Agent` wraps a provider with optional tools and memory, exposing `run()` for chat and `research()` for multi-round deep dives.

### Basic chat

```python
from kharness.providers import Ollama
from kharness.agent import Agent

provider = Ollama()
provider.set_default_model("qwen2.5:7b")

agent = Agent(provider)
response = agent.run("What is the capital of France?")
print(response)
```

### With a system prompt file

```python
agent = Agent(provider, soul_md_path="persona.md")
```

### With tools

Pass a dict or list of dicts mapping tool names to LangChain `@tool` objects:

```python
from kharness.tools import WEB_TOOL_MAP

agent = Agent(provider, tool_map=WEB_TOOL_MAP)
response = agent.run("What happened in the news today?")
```

### With memory

Attach a `Memory` object to maintain conversation history across turns:

```python
from kharness.memory import Memory

memory = Memory(max_turns=20)
agent = Agent(provider, memory=memory)

agent.run("My name is Alice.")
agent.run("What's my name?")   # remembers "Alice"
```

---

## Memory

`Memory` manages a sliding window of conversation history. It stores messages as `{"role": ..., "content": ...}` dicts — the format all providers expect.

```python
from kharness.memory import Memory

memory = Memory(max_turns=10)   # keeps last 10 user/assistant pairs
memory.add("user", "Hello")
memory.add("assistant", "Hi there!")

history = memory.get_history()  # list of message dicts
memory.clear()
```

| Parameter   | Default | Description                        |
| ----------- | ------- | ---------------------------------- |
| `max_turns` | `20`    | Max user/assistant pairs to retain |

---

## Research pipeline

`agent.research()` runs a structured multi-round research loop:

1. **Plan** — generates search queries from the question
2. **Gather** — runs web searches for each query
3. **Analyze** — synthesizes findings and decides if more searching is needed
4. **Repeat** — loops up to `max_rounds` times if gaps remain
5. **Report** — writes a final comprehensive answer

```python
from kharness.providers import Ollama
from kharness.tools import WEB_TOOL_MAP
from kharness.agent import Agent

provider = Ollama()
provider.set_default_model("qwen2.5:14b")

agent = Agent(provider, tool_map=WEB_TOOL_MAP)

report = agent.research(
    "What are the latest developments in fusion energy?",
    max_rounds=3,
    planning_tokens=512,
    analysis_tokens=2048,
    report_tokens=8000,
)
print(report)
```

| Parameter         | Default | Description                         |
| ----------------- | ------- | ----------------------------------- |
| `max_rounds`      | `3`     | Maximum search/analyze iterations   |
| `planning_tokens` | `1024`  | Token budget for query planning     |
| `analysis_tokens` | `2048`  | Token budget for per-round analysis |
| `report_tokens`   | `8000`  | Token budget for final report       |
| `debug`           | `True`  | Log phase progress                  |

---

## Tools

Tools are standard LangChain `@tool` functions. Convenience maps are exported for easy use with `Agent`.

### Web tools

```python
from kharness.tools import WEB_TOOL_MAP
```

| Tool                | Description                                  |
| ------------------- | -------------------------------------------- |
| `web_search`        | DuckDuckGo text search, returns top snippets |
| `fetch_page`        | Fetches a URL and extracts readable text     |
| `news_search`       | DuckDuckGo news search with dates            |
| `image_search`      | Returns direct image URLs                    |
| `wikipedia_summary` | Wikimedia REST API summary                   |
| `search_and_fetch`  | Combines search + full page fetch            |
| `search_subreddit`  | Fetches top posts from a subreddit           |

### Stock tools

```python
from kharness.tools import STOCK_TOOL_MAP
```

| Tool                    | Description                    |
| ----------------------- | ------------------------------ |
| `get_candles`           | OHLCV candlestick data         |
| `get_options`           | Options chain (calls and puts) |
| `get_income_statements` | Income statement financials    |
| `get_balance_sheet`     | Balance sheet financials       |
| `get_cash_flow`         | Cash flow statement            |

### Custom tools

Any LangChain `@tool` works:

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = Agent(provider, tool_map={"add": add})
```

---

## Configuration

Provider settings (model directory, default model) are persisted to `kharness/config/config.json` automatically via the `Config` class.

```python
provider.set_model_dir("/path/to/models")
provider.set_default_model("my-model.gguf")

print(provider.get_model_dir())
print(provider.get_default_model())
```

---

## Logging

KHarness uses Python's standard `logging` module. To see runtime output:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Use `logging.DEBUG` to see individual tool calls.

---

## Requirements

- Python >= 3.12
- A running [Ollama](https://ollama.com) daemon **or** [llama-server](https://github.com/ggerganov/llama.cpp) process
