#!/usr/bin/env python3
"""
Copy loop-baseline entries from old result files to new result files.

Usage:
    python copy_loop_baselines.py <task_number>
    python copy_loop_baselines.py 1 2 3 4 5 6  # Multiple tasks
    python copy_loop_baselines.py --all        # All tasks found

Examples:
    python copy_loop_baselines.py 4
    python copy_loop_baselines.py 1 2 3
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_OLD_DIR = SCRIPT_DIR / "results-old"
RESULTS_NEW_DIR = SCRIPT_DIR / "results"


def get_old_filename(task_num: int) -> Path:
    """Old files use 'dashdish-customN' naming (no hyphen before number)."""
    return RESULTS_OLD_DIR / f"dashdish-custom{task_num}_e2e_detailed.json"


def get_new_filename(task_num: int) -> Path:
    """New files use 'dashdish-custom-N' naming (hyphen before number)."""
    return RESULTS_NEW_DIR / f"dashdish-custom-{task_num}_e2e_detailed.json"


def copy_loop_baselines(task_num: int, dry_run: bool = False) -> bool:
    """
    Copy loop-baseline entries from old file to new file for a given task number.

    Returns True if successful, False otherwise.
    """
    old_file = get_old_filename(task_num)
    new_file = get_new_filename(task_num)

    print(f"\n{'=' * 60}")
    print(f"Task {task_num}")
    print(f"{'=' * 60}")
    print(f"Old file: {old_file}")
    print(f"New file: {new_file}")

    # Check if files exist
    if not old_file.exists():
        print(f"  ❌ Old file not found: {old_file}")
        return False

    if not new_file.exists():
        print(f"  ❌ New file not found: {new_file}")
        return False

    # Read both files
    try:
        with open(old_file, "r") as f:
            old_data = json.load(f)
        with open(new_file, "r") as f:
            new_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error: {e}")
        return False

    # Find loop-baseline entries (not loop-tools-baseline)
    loop_baseline_entries = []
    for entry in old_data.get("results", []):
        name = entry.get("name", "")
        if name.endswith("-loop-baseline"):
            loop_baseline_entries.append(entry)
            print(f"  Found: {name}")

    if not loop_baseline_entries:
        print("  ⚠️  No loop-baseline entries found in old file")
        return True

    # Get existing names in new file
    existing_names = [e.get("name", "") for e in new_data.get("results", [])]

    # Add entries that don't exist yet
    added = []
    skipped = []
    for entry in loop_baseline_entries:
        name = entry["name"]
        if name not in existing_names:
            if not dry_run:
                new_data["results"].append(entry)
            added.append(name)
            print(f"  ✅ Added: {name}")
        else:
            skipped.append(name)
            print(f"  ⏭️  Already exists: {name}")

    # Save the new file
    if added and not dry_run:
        with open(new_file, "w") as f:
            json.dump(new_data, f, indent=2)
        print(f"  💾 Saved {len(added)} new entries to {new_file.name}")
    elif dry_run and added:
        print(f"  🔍 [DRY RUN] Would add {len(added)} entries")

    return True


def find_all_task_numbers() -> list[int]:
    """Find all task numbers that have both old and new files."""
    task_nums = []

    # Look for old files
    for f in RESULTS_OLD_DIR.glob("dashdish-custom*_e2e_detailed.json"):
        # Extract number from filename like "dashdish-custom3_e2e_detailed.json"
        name = f.stem  # "dashdish-custom3_e2e_detailed"
        try:
            # Remove prefix and suffix to get the number
            num_str = name.replace("dashdish-custom", "").replace("_e2e_detailed", "")
            task_num = int(num_str)

            # Check if corresponding new file exists
            if get_new_filename(task_num).exists():
                task_nums.append(task_num)
        except ValueError:
            continue

    return sorted(task_nums)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable tasks with both old and new files:")
        for num in find_all_task_numbers():
            print(f"  - Task {num}")
        sys.exit(1)

    # Parse arguments
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if "--all" in sys.argv or "all" in args:
        task_nums = find_all_task_numbers()
        if not task_nums:
            print("No matching task files found!")
            sys.exit(1)
        print(f"Processing all {len(task_nums)} tasks: {task_nums}")
    else:
        try:
            task_nums = [int(arg) for arg in args]
        except ValueError:
            print(f"Error: Invalid task number(s). Please provide integers.")
            sys.exit(1)

    if dry_run:
        print("\n🔍 DRY RUN MODE - No files will be modified\n")

    # Process each task
    success = 0
    failed = 0
    for task_num in task_nums:
        if copy_loop_baselines(task_num, dry_run=dry_run):
            success += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary: {success} succeeded, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
