"""AI Pipeline Builder — turn a task description into a working HuggingFace pipeline."""

import argparse
import os
import sys
from pathlib import Path

import planner
import model_finder
import code_generator


def _format_downloads(downloads: int) -> str:
    if downloads >= 1_000_000:
        return f"{downloads / 1_000_000:.1f}M"
    elif downloads >= 1_000:
        return f"{downloads / 1_000:.1f}K"
    return str(downloads)


def _select_model_for_step(step: dict) -> dict:
    """Interactive model selection for a single pipeline step."""
    print(f"\n  Step {step['step']}: {step['description']}")
    print(f"    Task: {step['pipeline_tag']} | {step['input_type']} → {step['output_type']}")

    print(f"\n  Finding candidates...")
    candidates = model_finder.get_candidates_for_step(step)

    if not candidates:
        print(f"  No transformers-compatible models found for '{step['pipeline_tag']}'.")
        print(f"  Enter a custom HuggingFace model ID:")
        custom_id = input("  > ").strip()
        return model_finder.get_custom_model_info(custom_id, step)

    # Show ranked suggestions
    print(f"\n  Suggested models (ranked by fit):")
    for i, c in enumerate(candidates):
        dl = _format_downloads(c["downloads"])
        rec = " ← recommended" if i == 0 else ""
        print(f"    [{i + 1}] {c['model_id']} (↓ {dl} downloads){rec}")

    print(f"    [c] Enter a custom model ID")

    # Get user choice
    while True:
        choice = input(f"\n  Select model [1-{len(candidates)}/c] (default: 1): ").strip()

        if choice == "" or choice == "1":
            chosen = candidates[0]
            break
        elif choice.lower() == "c":
            custom_id = input("  Enter HuggingFace model ID (e.g. openai/whisper-large-v3): ").strip()
            return model_finder.get_custom_model_info(custom_id, step)
        elif choice.isdigit() and 1 <= int(choice) <= len(candidates):
            chosen = candidates[int(choice) - 1]
            break
        else:
            print(f"  Invalid choice. Enter 1-{len(candidates)} or 'c' for custom.")

    print(f"  ✓ Selected: {chosen['model_id']}")
    return {**step, **chosen}


def main():
    parser = argparse.ArgumentParser(
        description="Build an AI pipeline from a task description"
    )
    parser.add_argument("task", help="Describe the AI task you want to solve")
    parser.add_argument(
        "-o", "--output",
        default="output/pipeline.py",
        help="Output file path (default: output/pipeline.py)",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        sys.exit(1)

    task = args.task

    # Step 1: Plan the pipeline
    print("\n🔍 Analyzing task...\n")
    plan = planner.plan(task)
    steps = plan["steps"]

    print("📋 Pipeline Plan:")
    for step in steps:
        print(f"  Step {step['step']}: {step['description']}")
        print(f"    Task: {step['pipeline_tag']}")
        print(f"    Input: {step['input_type']} → Output: {step['output_type']}")

    # Step 2: Interactive model selection
    print("\n🔎 Model Selection")
    enriched_steps = []
    for step in steps:
        enriched = _select_model_for_step(step)
        enriched_steps.append(enriched)

    # Summary
    print("\n" + "=" * 50)
    print("📦 Final Pipeline:")
    for step in enriched_steps:
        dl = _format_downloads(step.get("downloads", 0))
        print(f"  Step {step['step']}: {step['description']}")
        print(f"    Model: {step['model_id']} (↓ {dl} downloads)")
    print("=" * 50)

    # Step 3: Generate code
    print("\n💻 Generating pipeline code...")
    code = code_generator.generate(task, enriched_steps)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)

    print(f"\n✅ Saved to: {output_path}")
    print(f"\nTo run:")
    print(f"  pip install transformers torch")
    print(f"  python {output_path}")


if __name__ == "__main__":
    main()
