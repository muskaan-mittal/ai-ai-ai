"""Decomposes a user task into ordered pipeline steps using OpenRouter."""

import json
from pathlib import Path

import hf_tasks
import llm

PROMPT_PATH = Path(__file__).parent / "prompts" / "planner.txt"


def _build_prompt(task_description: str, valid_tasks: list[str]) -> str:
    template = PROMPT_PATH.read_text()
    tags_str = ", ".join(sorted(valid_tasks))
    return template.replace("{valid_tags}", tags_str)


def _validate_plan(plan: dict, valid_tasks: list[str]) -> list[str]:
    """Returns a list of error messages. Empty list means valid."""
    errors = []
    steps = plan.get("steps", [])

    if not steps:
        errors.append("Plan has no steps.")
        return errors

    for step in steps:
        tag = step.get("pipeline_tag", "")
        if tag not in valid_tasks:
            errors.append(f"Step {step.get('step')}: invalid pipeline_tag '{tag}'.")

    for i in range(len(steps) - 1):
        current_out = steps[i].get("output_type", "")
        next_in = steps[i + 1].get("input_type", "")
        if current_out != next_in:
            errors.append(
                f"Step {steps[i]['step']} outputs '{current_out}' but "
                f"step {steps[i+1]['step']} expects '{next_in}'."
            )

    return errors


def plan(task_description: str) -> dict:
    """Decompose a task into pipeline steps. Returns the plan dict."""
    valid_tasks = hf_tasks.get_valid_tasks()
    system_prompt = _build_prompt(task_description, valid_tasks)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User's task: {task_description}"},
    ]

    for attempt in range(2):
        text = llm.chat(messages, temperature=0.2)

        try:
            result = llm.extract_json(text)
        except ValueError:
            if attempt == 0:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "Your previous response was not valid JSON. Please respond with ONLY valid JSON."})
                continue
            raise

        errors = _validate_plan(result, valid_tasks)
        if not errors:
            return result

        if attempt == 0:
            error_msg = "\n".join(errors)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Your previous plan had these errors:\n{error_msg}\nPlease fix them."})
            continue

        print(f"Warning: Plan has validation issues: {errors}")
        return result

    return result
