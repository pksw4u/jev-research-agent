# Jev Decision-Driven Research Agent

<img width="1037" height="952" alt="jev-mermaid" src="https://github.com/user-attachments/assets/2feb874f-c3f0-49e6-b0fe-60dc20133de6" />




An experimental research agent built on LangGraph where the division of labour
is explicit:

- **Reasoning & generation** → a general-purpose LLM (Config)
- **Decision making** → [TypeSafe Jev](https://typesafe.ai)
- **Execution** → the harness / tools

> **Jev decides what should happen. The harness executes the decision. The
> general LLM handles understanding, planning, and generation. Jev never calls
> a tool directly.**

## How it works

```
user request
    -> understand_task   (LLM: understanding, plan, candidate queries)
    -> select_action     (Jev: Choice over {search_web, search_news, finish})
        -> execute_tool  (harness runs the selected search)
            -> decide_continue (Jev: Noul — enough info?)
            -> loop back to select_action, or
    -> generate_answer   (LLM: synthesize findings)
```

Two Jev decisions drive the loop:

1. **Action selection** — a `Choice` question whose options are
   `search_web`, `search_news`, and `finish`, answered with a probability
   distribution and a confidence score.
2. **Sufficiency** — a `Noul` question ("enough information has been
   gathered to answer the request") whose probability is compared against a
   stop threshold.

All thresholds live in code (`app/decision/policy.py`), not in the model. If
Jev is missing an API key or a call fails, the layer degrades to a
deterministic policy fallback rather than crashing.

## Layout

```
app/
  main.py            CLI entry point
  graph/             state, nodes, LangGraph topology
  decision/          Jev layer, typed schemas, policy/thresholds/fallback
  llm/model.py       provider-agnostic chat model factory
  tools/             search_web / search_news (harness execution)
  prompts/           task understanding, planning, final answer
tests/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"            # or: pip install -r requirements.txt

cp .env.example .env               # then fill in the values
export TYPESAFE_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
```

Required: a general LLM API key (any OpenAI-compatible endpoint works, e.g. NVIDIA NIM).
Jev keys are early access; **without one the agent still runs** using the
deterministic policy fallback, which is handy for exercising the harness.

### Example: NVIDIA NIM (OpenAI-compatible)

```bash
export LLM_PROVIDER=nim
export LLM_MODEL=meta/llama-3.3-70b-instruct
export NVIDIA_API_KEY="nvapi-..."                  # https://build.nvidia.com/settings
# export LLM_BASE_URL="https://integrate.api.nvidia.com/v1"   # default anyway

python -m app.main "Did Acme raise funding recently?"
```

`LLM_BASE_URL` can point at any OpenAI-compatible endpoint — a self-hosted NIM
container, a local vLLM server (`http://localhost:8000/v1`), an OpenAI proxy,
or OpenRouter.

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | — | Jev key; absent ⇒ policy fallback |
| `TYPESAFE_MODEL` | `jev-latest` | Pin e.g. `jev-1.13.0` for stable thresholds |
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `google_genai`, `nim`, `openai_compatible`, … |
| `LLM_MODEL` | `gpt-4o-mini` | Model for the provider (NIM default: `meta/llama-3.3-70b-instruct`) |
| `LLM_BASE_URL` | — | OpenAI-compatible base URL override (NIM/vLLM/proxy) |
| `LLM_API_KEY` | — | Key override that wins over provider-specific keys |
| `OPENAI_API_KEY` | — | Auth for `openai` provider |
| `NVIDIA_API_KEY` | — | Auth for `nim` provider (https://build.nvidia.com/settings) |
| `SEARCH_PROVIDER` | auto | `tavily` or `duckduckgo`. Auto-selects Tavily when `TAVILY_API_KEY` is set, else DuckDuckGo |
| `TAVILY_API_KEY` | — | Needed for the tavily provider |
| `SEARCH_RESULTS` | `5` | Results per search |
| `MAX_RESEARCH_STEPS` | `4` | Loop cap before the agent is forced to finish |
| `ACTION_CONFIDENCE_THRESHOLD` | `0.6` | Jev action gate; below ⇒ policy fallback |
| `STOP_CONFIDENCE_THRESHOLD` | `0.7` | Noul probability needed to stop researching |
| `RECENT_OBSERVATIONS` | `4` | Most-recent findings fed back into Jev state |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | `jev-research-agent` | LangSmith project name |

## Run

```bash
python -m app.main "Find out whether Acme recently raised funding and summarize the findings."
```

### Search providers

Web and news search run through the harness (`app/tools/`). Provider selection
is: explicit `SEARCH_PROVIDER` → else Tavily if `TAVILY_API_KEY` is set → else
DuckDuckGo (no key needed). If a search fails (rate limit, auth, network), the
failure is recorded as an observation and the run continues rather than
crashing.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite runs fully offline: Jev and the LLM are faked, and search is
stubbed, so CI needs no keys.
