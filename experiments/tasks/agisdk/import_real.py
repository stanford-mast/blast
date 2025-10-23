"""
Imports REAL benchmark tasks from the agisdk repository (https://github.com/agi-inc/agisdk).
"""

import json
from pathlib import Path

import yaml

# Path to the cloned agisdk repo (expected to be a sibling folder named `agisdk`)
AGISDK_ROOT = Path(__file__).parent / Path("agisdk")
OUTPUT_DIR = Path(__file__).parent / Path("agisdk.yaml")


def _find_task_files():
    """Return a list of Path objects pointing to REAL task JSON files.

    Strategy:
    1. Try the original hard-coded path used historically (v2 flat tasks dir).
    2. Try common v1/v2 webclones tasks dirs under src.
    3. Fallback: scan the cloned repo for any JSON files located in a `tasks`
       directory under a path that contains `REAL`.
    4. Final fallback: scan all JSON files under the repo and keep only those
       that parse and contain a top-level "goal" key (likely a task file).
    """

    # Only use the v2 tasks directory as requested
    v2 = AGISDK_ROOT / Path("src/agisdk/REAL/browsergym/webclones/v2/tasks")
    if not v2.exists():
        print(f"v2 tasks directory not found: {v2}")
        return []

    return sorted(v2.glob("*.json"))


def main():
    output = []
    task_files = _find_task_files()
    if not task_files:
        print("No REAL task files found under the expected agisdk clone.\n"
              "If you haven't cloned the agisdk repo yet, run:\n"
              "  git clone https://github.com/agi-inc/agisdk.git experiments/tasks/agisdk/agisdk")
        return

    print(f"Found {len(task_files)} candidate task files; importing...")
    for task_file in task_files:
        try:
            if task_file.is_file():
                print(f"  {task_file}")
                with open(task_file, "r") as f:
                    data = json.load(f)

                output.append(
                    {
                        "id": data["id"],
                        "initial_url": data["website"]["url"],
                        "goal": data["goal"],
                    }
                )
        except Exception as e:
            print(f"Error loading task {task_file}: {e}")
            continue

    # Sort by id
    output.sort(key=lambda x: x["id"])

    with open(OUTPUT_DIR, "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    main()
