# AI Pipeline Builder — Detailed Implementation Plan

## TL;DR

User describes a task → we build them a working AI pipeline from HuggingFace models.

**5 pieces, in order:**

1. **Task Registry (`hf_tasks.py`)** — Hits the HuggingFace `/api/tasks` endpoint to get the list of all valid task types (image-to-text, text-generation, etc.) and what input/output types each one expects. This grounds everything in reality — we only plan with tasks that actually exist.

2. **Planner (`planner.py`)** — Sends the user's task description + the valid task list to Gemini. Gemini breaks the task into ordered steps, each tagged with a real HF pipeline tag. We validate that the steps chain correctly (step 1's output type matches step 2's input type).

3. **Model Finder (`model_finder.py`)** — For each step in the plan, searches the HuggingFace Hub for real models matching that task. Picks the most downloaded model that's `transformers`-compatible. Pulls structured metadata (`auto_model`, `processor` class names) — no model card parsing needed.

4. **Code Generator (`code_generator.py`)** — Sends the full enriched plan (steps + model IDs + metadata) to Gemini and asks it to produce a single runnable Python script using `transformers.pipeline()`. The metadata means Gemini doesn't have to guess how to call the models.

5. **CLI (`main.py`)** — Glues it all together. User runs `python main.py "my task"`, sees the plan, sees the selected models, gets a saved `.py` file they can run.

**Key insight:** We avoid the "how do you call each model" problem by (a) only selecting models that support the standard `transformers.pipeline()` API, and (b) pulling structured metadata from the HF Hub API (`transformers_info` field) so the code generator knows the exact classes to use.

---

## What We're Building

A Python CLI tool where a user describes a task in plain language, and the system:

1. Decomposes the task into ordered sub-tasks
2. Finds real HuggingFace models for each sub-task using the HF Hub API
3. Validates that the pipeline steps are compatible (output types match input types)
4. Generates a single runnable Python script that chains the models together

The user does not need to know which models exist, how they work, or how to connect them. They just describe what they want, and the tool gives them working code.

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| LLM (planning + code gen) | Gemini 2.5 Flash via `google-genai` SDK | Fast, cheap, good at structured JSON output |
| Model discovery | `huggingface_hub` Python library | Official SDK, gives structured model metadata |
| Task I/O specs | HuggingFace `/api/tasks` REST endpoint | Returns input/output type schemas per task |
| CLI interface | `argparse` (stdlib) | No extra dependency needed |
| Output | Single `.py` file saved to disk | User gets one runnable file, no framework to learn |

---

## File Structure

```
ai-ai-ai/
├── main.py              # CLI entry point — ties everything together
├── planner.py           # Gemini decomposes user task into pipeline steps
├── model_finder.py      # Searches HF Hub for real models per step
├── code_generator.py    # Gemini generates runnable pipeline code
├── hf_tasks.py          # Fetches + caches /api/tasks for I/O type info
├── prompts/
│   ├── planner.txt      # System prompt for task decomposition
│   └── codegen.txt      # System prompt for code generation
├── output/              # Where generated pipeline scripts are saved
├── requirements.txt     # google-genai, huggingface_hub, requests
├── idea.md              # Original idea document
└── PLAN.md              # This file
```

---

## Detailed Flow

### Overview

```
User Task Description (string)
        │
        ▼
┌──────────────────────────────────────────────────┐
│  hf_tasks.py                                      │
│  Fetch valid HF pipeline tags + I/O type schemas  │
│  from https://huggingface.co/api/tasks            │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  planner.py                                       │
│  Send task + valid tags to Gemini                 │
│  → Returns ordered steps with pipeline_tags       │
│  → Validate tags exist and I/O types chain        │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  model_finder.py                                  │
│  For each step, search HF Hub by pipeline_tag     │
│  → Pick top model with transformers support        │
│  → Extract auto_model, processor, widget_data     │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  code_generator.py                                │
│  Send enriched steps to Gemini                    │
│  → Returns a complete Python script               │
│  → Save to output/pipeline.py                     │
└──────────────────────────────────────────────────┘
```

---

### Step 1: HF Task Registry (`hf_tasks.py`)

**Purpose:** Fetch and cache the list of valid HuggingFace tasks with their input/output type schemas. This runs once at the start and provides data used by both the planner (to constrain valid tags) and the validator (to check I/O compatibility).

**What it does:**

1. GET `https://huggingface.co/api/tasks`
2. Parse out for each task: `id`, `input types`, `output types` (from the `demo.inputs` and `demo.outputs` fields)
3. Cache in memory (fetched once per run, no disk caching needed)

**Public functions:**

```python
def get_valid_tasks() -> list[str]:
    """Returns list of valid pipeline_tag strings, e.g. ['text-generation', 'image-to-text', ...]"""

def get_task_io(pipeline_tag: str) -> dict:
    """Returns I/O schema for a task, e.g. {"inputs": ["image"], "outputs": ["text"]}"""
```

**Example data from the API for one task:**

```json
{
  "image-to-text": {
    "id": "image-to-text",
    "label": "Image to Text",
    "demo": {
      "inputs": [{"type": "img", "label": "Input"}],
      "outputs": [{"type": "text", "label": "Output"}]
    }
  }
}
```

We extract and normalize the types: `"img"` → `"image"`, `"text"` → `"text"`, `"audio"` → `"audio"`.

---

### Step 2: Task Decomposition (`planner.py`)

**Input:** User's plain-language task description (string) + list of valid HF tasks from `hf_tasks.py`

**What happens:**

1. Build the prompt by injecting the valid task list into `prompts/planner.txt`
2. Send to Gemini 2.5 Flash with `response_mime_type="application/json"` for structured output
3. Parse the JSON response
4. Validate:
   - Every `pipeline_tag` in the response is a real HF task (exists in the valid task list)
   - The I/O types chain correctly: step N's `output_type` matches step N+1's `input_type`
   - If validation fails, retry once with the error message appended to the prompt

**Gemini call:**

```python
from google import genai
import os, json

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "temperature": 0.2  # low temperature for deterministic planning
    }
)

plan = json.loads(response.text)
```

**Output format (JSON):**

```json
{
  "steps": [
    {
      "step": 1,
      "description": "Generate a caption describing the input image",
      "pipeline_tag": "image-to-text",
      "input_type": "image",
      "output_type": "text"
    },
    {
      "step": 2,
      "description": "Generate an answer combining the caption with the user's question",
      "pipeline_tag": "text2text-generation",
      "input_type": "text",
      "output_type": "text"
    }
  ]
}
```

---

### Step 3: Model Discovery (`model_finder.py`)

**Input:** The list of steps from the planner, each with a `pipeline_tag`

**What happens for each step:**

1. Call `huggingface_hub.list_models(task=pipeline_tag, sort="downloads", limit=5)` to get the top 5 models by download count
2. For each candidate model, fetch detailed info via `huggingface_hub.model_info(model_id)` to get:
   - `library_name` — must be `"transformers"` (skip models using diffusers, sentence-transformers, etc. for v1)
   - `transformers_info.auto_model` — the exact Auto class (e.g., `"AutoModelForVision2Seq"`)
   - `transformers_info.processor` — the processor class (e.g., `"AutoProcessor"`, `"AutoTokenizer"`)
   - `downloads` — for ranking
   - `widget_data` — example inputs for the generated test code
3. Pick the first model that satisfies: `library_name == "transformers"` AND `transformers_info` is not None

**Code sketch:**

```python
from huggingface_hub import list_models, model_info

def find_model_for_step(step: dict) -> dict:
    candidates = list_models(task=step["pipeline_tag"], sort="downloads", limit=5)
    
    for candidate in candidates:
        info = model_info(candidate.id)
        if info.library_name == "transformers" and info.transformers_info:
            return {
                **step,
                "model_id": info.id,
                "auto_model": info.transformers_info.auto_model,
                "processor": info.transformers_info.processor,
                "downloads": info.downloads,
                "widget_data": info.widget_data or []
            }
    
    # Fallback: return the most downloaded model even without transformers_info
    return {**step, "model_id": candidates[0].id, "auto_model": None, "processor": None}
```

**Output format (enriched step):**

```json
{
  "step": 1,
  "description": "Generate a caption describing the input image",
  "pipeline_tag": "image-to-text",
  "input_type": "image",
  "output_type": "text",
  "model_id": "Salesforce/blip-image-captioning-large",
  "auto_model": "AutoModelForVision2Seq",
  "processor": "AutoProcessor",
  "downloads": 2100000,
  "widget_data": [{"src": "https://cdn-media.huggingface.co/.../example.jpg"}]
}
```

---

### Step 4: Code Generation (`code_generator.py`)

**Input:** The user's original task description + the enriched pipeline steps (with model IDs and metadata)

**What happens:**

1. Build a detailed prompt from `prompts/codegen.txt` containing:
   - The user's original task description
   - Each step with: model ID, auto_model class, processor class, pipeline_tag
   - Instructions for code style and structure (see prompt below)
2. Send to Gemini 2.5 Flash
3. Extract the Python code from the response (strip markdown fences if present)
4. Return the code as a string

**Gemini call:**

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "temperature": 0.1  # very low temperature for deterministic code
    }
)

code = response.text
# Strip markdown code fences if present
if code.startswith("```python"):
    code = code[len("```python"):].rsplit("```", 1)[0]
```

**Output:** A string containing the complete Python script.

---

### Step 5: CLI Glue (`main.py`)

**The user runs:**

```bash
python main.py "I want an AI that looks at an image and answers questions about it"
```

**What main.py does, in order:**

1. Parse the CLI argument using `argparse`
2. Call `hf_tasks.get_valid_tasks()` to fetch the list of valid HF pipeline tags
3. Call `planner.plan(task_description, valid_tasks)` → get pipeline steps as JSON
4. Print the plan to the terminal for user visibility
5. Call `model_finder.find_models(steps)` → get enriched steps with real model IDs
6. Print the selected models with download counts
7. Call `code_generator.generate(task_description, enriched_steps)` → get Python code string
8. Create `output/` directory if it doesn't exist
9. Save the code to `output/pipeline.py`
10. Print the file path and pip install instructions

**Example terminal output:**

```
$ python main.py "Look at an image and answer questions about it using external knowledge"

🔍 Analyzing task...

📋 Pipeline Plan:
  Step 1: Image Captioning
    Task: image-to-text
    Input: image → Output: text

  Step 2: Answer Generation
    Task: text2text-generation
    Input: text → Output: text

🔎 Finding models...
  Step 1: Salesforce/blip-image-captioning-large (↓ 2.1M downloads)
  Step 2: google/flan-t5-large (↓ 4.8M downloads)

💻 Generating pipeline code...

✅ Saved to: output/pipeline.py

To run:
  pip install transformers torch Pillow
  python output/pipeline.py
```

---

## Prompt Design

### `prompts/planner.txt`

```
You are an AI pipeline architect. Given a user's task description, decompose it
into a sequence of steps where each step can be handled by a single HuggingFace
model.

For each step, assign one of these valid HuggingFace pipeline tags:
{valid_tags}

Rules:
- Use the minimum number of steps needed. Do not add unnecessary steps.
- Each step must use exactly one pipeline_tag from the list above.
- The output type of step N must be compatible with the input type of step N+1.
  For example, if step 1 outputs "text", step 2 must accept "text" as input.
- Common type flows: image→text, text→text, audio→text, text→image, text→audio.
- If the task can be done with a single model, use a single step.
- If the task requires multiple capabilities (e.g., understanding an image AND
  generating text), use multiple steps.

Respond with ONLY valid JSON in this exact format:
{
  "steps": [
    {
      "step": 1,
      "description": "A clear description of what this step does",
      "pipeline_tag": "one-of-the-valid-tags",
      "input_type": "text|image|audio",
      "output_type": "text|image|audio"
    }
  ]
}
```

### `prompts/codegen.txt`

```
You are a Python code generator specializing in HuggingFace Transformers pipelines.

Generate a single, self-contained Python script that implements the following AI pipeline.

User's task: {task_description}

Pipeline steps:
{steps_json}

Rules:
- Use `from transformers import pipeline` for each model step.
- For each step, initialize the pipeline like this:
    step_N = pipeline("{pipeline_tag}", model="{model_id}")
- Wire the output of each step into the input of the next step.
- Handle the data format conversion between steps. For example:
  - If a pipeline returns a list of dicts like [{"generated_text": "..."}], extract the string.
  - If a pipeline returns [{"label": "...", "score": 0.9}], format appropriately.
- Include a comment block at the very top listing all required pip packages:
    # Requirements: pip install transformers torch Pillow
- Include a main() function that demonstrates the full pipeline with example inputs.
- Use if __name__ == "__main__": main() at the bottom.
- Include print statements showing the output of each step so the user can see the flow.
- Keep the code simple, readable, and well-commented.
- Do NOT include any explanation outside the code. Return ONLY the Python code.
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM provider | Gemini 2.5 Flash | User preference; fast and cheap for structured output |
| Model filtering | `library_name == "transformers"` only | Ensures `pipeline()` API works consistently; diffusers/sentence-transformers have different calling conventions |
| Model ranking | Sort by downloads, pick top match | Popular models are battle-tested, well-documented, and more likely to work out of the box |
| Code generation style | `transformers.pipeline()` API | Simplest, most universal calling convention across all task types; avoids needing model-specific code |
| Model metadata source | `transformers_info` from HF Hub API | Gives us exact class names without parsing model cards; reliable structured data |
| Model card fetching | Not used in v1 | `transformers_info` + `pipeline_tag` is sufficient for `pipeline()` based code; model cards are expensive (2k-10k tokens each) |
| Structured JSON from Gemini | `response_mime_type="application/json"` | Guarantees valid JSON response; no regex parsing needed |
| CLI over web UI | `argparse` | Zero setup, no server, no frontend — simplest possible UX for a demo |
| Output format | Single `.py` file | User gets one file they can read, modify, and run — no framework lock-in |

---

## HuggingFace API Details

### Model metadata fields we use

These are the specific fields from `huggingface_hub.model_info()` that our tool relies on:

| Field | Type | Example | Used For |
|---|---|---|---|
| `id` | `str` | `"Salesforce/blip-image-captioning-large"` | Model identifier in generated code |
| `pipeline_tag` | `str` | `"image-to-text"` | Matching models to pipeline steps |
| `library_name` | `str` | `"transformers"` | Filtering to transformers-compatible models |
| `transformers_info.auto_model` | `str` | `"AutoModelForVision2Seq"` | Informing the LLM about the model's class |
| `transformers_info.processor` | `str` | `"AutoProcessor"` | Informing the LLM about the tokenizer/processor |
| `downloads` | `int` | `2100000` | Ranking models by popularity |
| `widget_data` | `list[dict]` | `[{"src": "https://...jpg"}]` | Example inputs for generated test code |

### `/api/tasks` endpoint

- URL: `https://huggingface.co/api/tasks`
- Returns: JSON object with all ~45 task types
- Key fields per task: `id`, `demo.inputs[].type`, `demo.outputs[].type`
- Type values: `"text"`, `"img"`, `"audio"`, `"chart"`, `"tabular"`
- We normalize: `"img"` → `"image"` for consistency

---

## Dependencies

### Our tool's dependencies (`requirements.txt`):

```
google-genai
huggingface_hub
requests
```

Three packages. That's it. `requests` may already be pulled in by the others but we list it explicitly since `hf_tasks.py` uses it directly for the `/api/tasks` call.

### Generated code's dependencies (included as comments in output):

These vary by pipeline but commonly include:

```
transformers
torch
Pillow          # for image tasks
soundfile       # for audio tasks
sentencepiece   # for some tokenizers
```

---

## Error Handling

| Error | How We Handle It |
|---|---|
| Gemini returns invalid JSON | Retry once; if still invalid, print error and exit |
| Planner uses an invalid `pipeline_tag` | Validate against `/api/tasks` list; retry with correction prompt |
| I/O types don't chain (e.g., image→audio) | Detect mismatch, retry planner with explicit error |
| No transformers-compatible model found for a task | Fall back to the most downloaded model regardless of library; warn user |
| `GEMINI_API_KEY` not set | Print clear error message and exit |
| HuggingFace API is down or rate-limited | Print error and exit (no retry for v1) |

---

## Example End-to-End Walkthrough

**User runs:**

```bash
python main.py "I have a collection of product images and I want to automatically generate alt-text descriptions for accessibility"
```

**Step 1 — Planner output:**

```json
{
  "steps": [
    {
      "step": 1,
      "description": "Generate a natural language caption for the product image",
      "pipeline_tag": "image-to-text",
      "input_type": "image",
      "output_type": "text"
    }
  ]
}
```

(Single step — the planner correctly identifies this only needs one model.)

**Step 2 — Model finder output:**

```json
{
  "step": 1,
  "description": "Generate a natural language caption for the product image",
  "pipeline_tag": "image-to-text",
  "model_id": "Salesforce/blip-image-captioning-large",
  "auto_model": "AutoModelForVision2Seq",
  "processor": "AutoProcessor",
  "downloads": 2100000,
  "widget_data": [{"src": "https://cdn-media.huggingface.co/.../example.jpg"}]
}
```

**Step 3 — Generated code (`output/pipeline.py`):**

```python
# Requirements: pip install transformers torch Pillow
# Generated by AI Pipeline Builder
# Task: Generate alt-text descriptions for product images

from transformers import pipeline
from PIL import Image

def main():
    # Step 1: Generate a natural language caption for the product image
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    # Load an example image
    image = Image.open("your_image.jpg")

    # Run the pipeline
    result = captioner(image)

    # Extract the generated caption
    caption = result[0]["generated_text"]
    print(f"Generated alt-text: {caption}")

if __name__ == "__main__":
    main()
```

---

## What's Out of Scope for v1

- Web UI or visual pipeline builder
- Models that aren't `transformers`-compatible (diffusers, sentence-transformers, spaCy, etc.)
- Training or fine-tuning any models
- Actually running the generated pipeline (user does that themselves)
- Multi-input pipelines (e.g., two images at once, or image + separate text file)
- Comparing alternative pipeline designs
- Model card fetching as fallback for non-standard models
- GPU/memory estimation or hardware requirements
- Authentication for gated models (e.g., Llama)
- Caching or persisting results between runs
- Video or document (PDF) inputs
- Streaming or real-time pipelines

---

## Build Order

Recommended order for implementation:

1. **`hf_tasks.py`** — Fetch and parse `/api/tasks` (standalone, no dependencies on other modules)
2. **`prompts/planner.txt` and `prompts/codegen.txt`** — Write the prompt templates
3. **`planner.py`** — Wire up Gemini for task decomposition + validation
4. **`model_finder.py`** — Wire up HuggingFace Hub search + model selection
5. **`code_generator.py`** — Wire up Gemini for code generation
6. **`main.py`** — Tie everything together with CLI
7. **`requirements.txt`** — List dependencies
8. **Test with 2-3 example tasks** — e.g., image captioning, document QA, text summarization
