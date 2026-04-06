"""Searches HuggingFace Hub for real models matching each pipeline step."""

import json

from huggingface_hub import list_models, model_info as get_model_info

import llm


def _rank_candidates(step: dict, candidates: list[dict]) -> list[dict]:
    """Use LLM to rank candidates by relevance to the task. Returns reordered list."""
    candidate_summary = [
        {"model_id": c["model_id"], "downloads": c["downloads"], "tags": c.get("tags", [])}
        for c in candidates
    ]

    prompt = f"""Given this pipeline step:
- Description: {step['description']}
- Task type: {step['pipeline_tag']}
- Input: {step['input_type']} → Output: {step['output_type']}

Rank these models from best to worst fit. Prefer general-purpose English models unless the task clearly requires another language. Prefer models with high downloads that are well-known and reliable.

Candidates:
{json.dumps(candidate_summary, indent=2)}

Respond with ONLY a JSON object: {{"ranked_ids": ["best-model-id", "second-best", ...]}}"""

    text = llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
    result = llm.extract_json(text)
    ranked_ids = result.get("ranked_ids", [])

    id_to_candidate = {c["model_id"]: c for c in candidates}
    ranked = [id_to_candidate[mid] for mid in ranked_ids if mid in id_to_candidate]
    for c in candidates:
        if c["model_id"] not in ranked_ids:
            ranked.append(c)
    return ranked


def get_candidates_for_step(step: dict) -> list[dict]:
    """Get ranked list of transformers-compatible model candidates for a step."""
    tag = step["pipeline_tag"]

    candidates = list(list_models(pipeline_tag=tag, sort="downloads", limit=10))

    valid_candidates = []
    for candidate in candidates:
        try:
            info = get_model_info(candidate.id)
        except Exception:
            continue

        if info.library_name == "transformers" and info.transformers_info:
            valid_candidates.append({
                "model_id": info.id,
                "auto_model": info.transformers_info.auto_model,
                "processor": info.transformers_info.processor,
                "downloads": info.downloads,
                "widget_data": info.widget_data or [],
                "tags": info.tags or [],
            })

    if not valid_candidates:
        return []

    if len(valid_candidates) > 1:
        valid_candidates = _rank_candidates(step, valid_candidates)

    return valid_candidates


def get_custom_model_info(model_id: str, step: dict) -> dict:
    """Fetch metadata for a user-specified custom model."""
    try:
        info = get_model_info(model_id)
    except Exception as e:
        print(f"  Warning: Could not fetch info for '{model_id}': {e}")
        return {
            **step,
            "model_id": model_id,
            "auto_model": None,
            "processor": None,
            "downloads": 0,
            "widget_data": [],
        }

    return {
        **step,
        "model_id": info.id,
        "auto_model": info.transformers_info.auto_model if info.transformers_info else None,
        "processor": info.transformers_info.processor if info.transformers_info else None,
        "downloads": info.downloads or 0,
        "widget_data": info.widget_data or [],
    }
