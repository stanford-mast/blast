"""
Imports REAL benchmark tasks from the agisdk repository (https://github.com/agi-inc/agisdk).
"""

from pathlib import Path
import json
import yaml

# Path to the REAL tasks directory
REAL_TASK_DIR = Path(__file__).parent / Path(
    "agisdk/src/agisdk/REAL/browsergym/webclones/tasks"
)
OUTPUT_DIR = Path(__file__).parent / Path("agisdk.yaml")


def main():
    output = []
    for task_file in REAL_TASK_DIR.iterdir():
        try:
            if task_file.is_file():
                print(task_file.name)
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
            print(f"Error loading task {task_file.name}: {e}")
            continue

    # Sort by id
    output.sort(key=lambda x: x["id"])

    with open(OUTPUT_DIR, "w") as f:
        yaml.dump(
            output, f, sort_keys=False, default_flow_style=False, allow_unicode=True
        )


if __name__ == "__main__":
    main()
