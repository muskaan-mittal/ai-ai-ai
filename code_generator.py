"""Generates a runnable Python script from enriched pipeline steps using OpenRouter."""

import json
from pathlib import Path

from transformers import pipelines as _pipelines

import llm

PROMPT_PATH = Path(__file__).parent / "prompts" / "codegen.txt"


def _get_valid_pipeline_tasks() -> list[str]:
    """Get the list of tasks actually supported by transformers.pipeline()."""
    try:
        return list(_pipelines.PIPELINE_REGISTRY.get_supported_tasks())
    except Exception:
        return [
            "any-to-any", "audio-classification", "automatic-speech-recognition",
            "depth-estimation", "document-question-answering", "feature-extraction",
            "fill-mask", "image-classification", "image-feature-extraction",
            "image-segmentation", "image-text-to-text", "mask-generation",
            "object-detection", "sentiment-analysis", "table-question-answering",
            "text-classification", "text-generation", "text-to-audio",
            "text-to-speech", "token-classification", "video-classification",
            "zero-shot-audio-classification", "zero-shot-classification",
            "zero-shot-image-classification", "zero-shot-object-detection",
        ]


def _build_prompt(task_description: str, steps: list[dict]) -> str:
    template = PROMPT_PATH.read_text()

    clean_steps = []
    for s in steps:
        clean_steps.append({
            "step": s["step"],
            "description": s["description"],
            "pipeline_tag": s["pipeline_tag"],
            "model_id": s["model_id"],
            "auto_model": s.get("auto_model"),
            "processor": s.get("processor"),
            "input_type": s["input_type"],
            "output_type": s["output_type"],
        })

    valid_tasks = _get_valid_pipeline_tasks()

    prompt = template.replace("{task_description}", task_description)
    prompt = prompt.replace("{steps_json}", json.dumps(clean_steps, indent=2))
    prompt = prompt.replace("{valid_pipeline_tasks}", ", ".join(sorted(valid_tasks)))
    return prompt


def generate(task_description: str, steps: list[dict]) -> str:
    """Generate a Python pipeline script. Returns the code as a string."""
    prompt = _build_prompt(task_description, steps)
    text = llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
    return llm.extract_code(text)
