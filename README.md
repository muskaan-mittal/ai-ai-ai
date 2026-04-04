# Jac Multi-Agent Pipeline — Full-Stack Example

A complete example showing how to build a **multi-agent AI pipeline** in
**Jac/Jaseci** with a polished frontend.

```
User Input (Frontend)
      │
      ▼
┌─────────────────────────────────────────────────┐
│              Jac Backend  (main.jac)            │
│                                                 │
│  IntakeNode ──► ResearchNode ──► ReasoningNode  │
│      │               │                │         │
│  IntakeAgent    ResearchAgent   ReasoningAgent  │
│                                        │         │
│                               SummaryNode        │
│                                   │              │
│                              SummaryAgent        │
└─────────────────────────────────────────────────┘
      │
      ▼
Final Answer + Full Agent Trace (JSON)
```

## Jac Concepts Used

| Concept | Where |
|---|---|
| `node` | `PipelineRoot`, `IntakeNode`, `ResearchNode`, `ReasoningNode`, `SummaryNode` |
| `edge` | `intake_to_research`, `research_to_reasoning`, `reasoning_to_summary` |
| `walker` | `IntakeAgent`, `ResearchAgent`, `ReasoningAgent`, `SummaryAgent` |
| `spawn` | Each agent spawned onto its corresponding node |
| `visit [-->]` | Walkers traverse the graph automatically |
| `:can:` | Module-level functions (`run_pipeline`, `build_pipeline`, `call_claude`) |

## Project Structure

```
jaclang_app/
├── main.jac      ← Jac backend (graph + walkers + FastAPI server)
├── index.html    ← Frontend single-page app
└── README.md     ← This file
```

## Setup

### 1. Install dependencies

```bash
pip install jaclang fastapi uvicorn anthropic
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Start the backend

```bash
jac run main.jac
# Server starts at http://localhost:8000
```

### 4. Open the frontend

Open `index.html` in your browser (or serve it):

```bash
python -m http.server 3000
# Then visit http://localhost:3000
```

## API

### POST /api/query

```json
{ "query": "What causes the northern lights?" }
```

**Response (success):**
```json
{
  "success": true,
  "final_answer": "The northern lights are caused by...",
  "agent_trace": [
    { "agent": "IntakeAgent",    "is_valid": true,  "category": "question", "intent": "..." },
    { "agent": "ResearchAgent",  "context": "...",  "key_facts": ["..."] },
    { "agent": "ReasoningAgent", "analysis": "...", "confidence": "high" },
    { "agent": "SummaryAgent",   "final_answer": "..." }
  ]
}
```

**Response (blocked by Intake Agent):**
```json
{
  "success": false,
  "error": "Input was flagged as invalid or harmful.",
  "agent_trace": [{ "agent": "IntakeAgent", ... }]
}
```

## How the Pipeline Works

1. **IntakeAgent** — Validates and classifies the user input, extracting intent.
   If invalid/harmful, the pipeline short-circuits here.

2. **ResearchAgent** — Given the intent, surfaces background context and 3–5 key facts.

3. **ReasoningAgent** — Takes the research output and reasons through the problem,
   producing an analysis with a confidence level.

4. **SummaryAgent** — Synthesizes everything into a clean, conversational final answer.

The frontend shows each agent's output as animated cards with a live pipeline-stage indicator.
# ai-ai-ai
