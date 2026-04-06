"""Fetches and caches HuggingFace task definitions from /api/tasks."""

import requests

_cache = None

TYPE_NORMALIZE = {
    "img": "image",
    "text": "text",
    "audio": "audio",
    "chart": "text",
    "tabular": "text",
}


def _fetch_tasks() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    resp = requests.get("https://huggingface.co/api/tasks", timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    tasks = {}
    for task_id, task_data in raw.items():
        # Skip non-dict entries (the API sometimes includes metadata keys)
        if not isinstance(task_data, dict):
            continue

        demo = task_data.get("demo")
        if not demo:
            # Still register the task but with unknown I/O
            tasks[task_id] = {"inputs": [], "outputs": []}
            continue

        inputs = []
        for inp in demo.get("inputs", []):
            t = TYPE_NORMALIZE.get(inp.get("type", ""), inp.get("type", ""))
            if t and t not in inputs:
                inputs.append(t)

        outputs = []
        for out in demo.get("outputs", []):
            t = TYPE_NORMALIZE.get(out.get("type", ""), out.get("type", ""))
            if t and t not in outputs:
                outputs.append(t)

        tasks[task_id] = {"inputs": inputs, "outputs": outputs}

    _cache = tasks
    return _cache


def get_valid_tasks() -> list[str]:
    """Returns list of valid pipeline_tag strings."""
    return list(_fetch_tasks().keys())


def get_task_io(pipeline_tag: str) -> dict:
    """Returns I/O schema for a task, e.g. {"inputs": ["image"], "outputs": ["text"]}."""
    tasks = _fetch_tasks()
    return tasks.get(pipeline_tag, {"inputs": [], "outputs": []})
