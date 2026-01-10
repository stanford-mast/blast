#!/usr/bin/env python3
"""
Validate tool constraints for consistency and contradictions.

Checks:
1. Circular pre_tools dependencies
2. Unreachable tools (pre-conditions never satisfiable)
3. State flow gaps (page_type in pre that no post produces)
4. Dead-end states (post from which no progress possible)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def load_tools(path: str) -> List[Dict[str, Any]]:
    """Load tools from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("tools", [])


def check_circular_pretools(tools: List[Dict[str, Any]]) -> List[str]:
    """Check for circular pre_tools dependencies."""
    errors = []

    # Build dependency graph
    deps: Dict[str, Set[str]] = {}
    for tool in tools:
        name = tool["name"]
        pre_tools = tool.get("pre_tools", {})
        required = set()
        for tool_list in pre_tools.values():
            required.update(tool_list)
        deps[name] = required

    # Check for cycles using DFS
    def has_cycle(
        node: str, visited: Set[str], path: Set[str]
    ) -> Tuple[bool, List[str]]:
        visited.add(node)
        path.add(node)

        for dep in deps.get(node, set()):
            if dep in path:
                return True, list(path) + [dep]
            if dep not in visited:
                cycle_found, cycle_path = has_cycle(dep, visited, path)
                if cycle_found:
                    return True, cycle_path

        path.remove(node)
        return False, []

    visited: Set[str] = set()
    for tool_name in deps:
        if tool_name not in visited:
            cycle_found, cycle_path = has_cycle(tool_name, visited, set())
            if cycle_found:
                errors.append(f"Circular dependency: {' -> '.join(cycle_path)}")

    return errors


def check_state_flow(
    tools: List[Dict[str, Any]], initial_state: Dict[str, Any]
) -> List[str]:
    """Check for unreachable tools and state flow gaps."""
    errors = []

    # Collect all page_types that can be produced (from post conditions)
    producible_states: Dict[str, Set[str]] = defaultdict(set)

    # Add initial state
    for key, value in initial_state.items():
        if value and value != "*":
            producible_states[key].add(value)

    # Add states from all post conditions
    for tool in tools:
        post = tool.get("post", {})
        for key, value in post.items():
            if value and value != "*" and not value.startswith("$"):
                producible_states[key].add(value)

    # Check each tool's pre-conditions
    for tool in tools:
        name = tool["name"]
        pre = tool.get("pre", {})

        for key, required_value in pre.items():
            if required_value == "*" or required_value == "":
                continue

            # Check if this required value can ever be produced
            if key not in producible_states:
                errors.append(
                    f"Tool '{name}' requires {key}='{required_value}' but no tool produces this state key"
                )
            elif required_value not in producible_states[key]:
                available = producible_states[key]
                errors.append(
                    f"Tool '{name}' requires {key}='{required_value}' but available values are: {available}"
                )

    return errors


def check_pretools_exist(tools: List[Dict[str, Any]]) -> List[str]:
    """Check that all pre_tools references exist."""
    errors = []
    tool_names = {t["name"] for t in tools}

    for tool in tools:
        name = tool["name"]
        pre_tools = tool.get("pre_tools", {})

        for param, required_tools in pre_tools.items():
            for req_tool in required_tools:
                if req_tool not in tool_names:
                    errors.append(
                        f"Tool '{name}' requires non-existent tool '{req_tool}' in pre_tools"
                    )

    return errors


def check_dead_ends(tools: List[Dict[str, Any]]) -> List[str]:
    """Check for tools that produce states from which no progress is possible."""
    warnings = []

    # Build map of what pre-conditions each tool needs
    tool_pres: Dict[str, Dict[str, str]] = {}
    for tool in tools:
        tool_pres[tool["name"]] = tool.get("pre", {})

    # For each tool, check if its post allows any other tool to be called
    for tool in tools:
        name = tool["name"]
        post = tool.get("post", {})

        if not post:
            continue

        # Check if any other tool can be called after this one
        can_continue = False
        for other_tool in tools:
            if other_tool["name"] == name:
                continue

            other_pre = other_tool.get("pre", {})
            if not other_pre:
                can_continue = True
                break

            # Check if post satisfies other_pre
            matches = True
            for key, required in other_pre.items():
                if required == "*":
                    continue
                post_value = post.get(key)
                if post_value and post_value != "*" and not post_value.startswith("$"):
                    if post_value != required:
                        matches = False
                        break

            if matches:
                can_continue = True
                break

        if not can_continue:
            # Check if this is a terminal tool (order confirmation, etc.)
            post_page = post.get("page_type", "")
            if post_page not in ["order_confirmation"]:
                warnings.append(f"Tool '{name}' may produce a dead-end state: {post}")

    return warnings


def check_type_consistency(tools: List[Dict[str, Any]]) -> List[str]:
    """Check that tool types match their behavior."""
    warnings = []

    for tool in tools:
        name = tool["name"]
        tool_type = tool.get("type", "")
        pre = tool.get("pre", {})
        post = tool.get("post", {})

        # gotoItem should change page_type
        if tool_type == "gotoItem":
            pre_page = pre.get("page_type")
            post_page = post.get("page_type")
            if pre_page and post_page and pre_page == post_page and pre_page != "*":
                # Only warn if they're both concrete and the same
                if not post_page.startswith("$"):
                    warnings.append(
                        f"Tool '{name}' is type 'gotoItem' but doesn't change page_type "
                        f"(pre={pre_page}, post={post_page})"
                    )

    return warnings


def validate_tools(path: str, initial_state: Dict[str, Any] = None) -> bool:
    """Run all validations on a tools file."""
    if initial_state is None:
        initial_state = {"page_type": "home"}

    print(f"Validating: {path}")
    print("=" * 60)

    tools = load_tools(path)
    print(f"Loaded {len(tools)} tools\n")

    all_errors = []
    all_warnings = []

    # Run checks
    print("Checking circular pre_tools dependencies...")
    errors = check_circular_pretools(tools)
    all_errors.extend(errors)
    print(f"  Found {len(errors)} errors\n")

    print("Checking pre_tools references exist...")
    errors = check_pretools_exist(tools)
    all_errors.extend(errors)
    print(f"  Found {len(errors)} errors\n")

    print("Checking state flow consistency...")
    errors = check_state_flow(tools, initial_state)
    all_errors.extend(errors)
    print(f"  Found {len(errors)} errors\n")

    print("Checking for dead-end states...")
    warnings = check_dead_ends(tools)
    all_warnings.extend(warnings)
    print(f"  Found {len(warnings)} warnings\n")

    print("Checking type consistency...")
    warnings = check_type_consistency(tools)
    all_warnings.extend(warnings)
    print(f"  Found {len(warnings)} warnings\n")

    # Report results
    print("=" * 60)

    if all_errors:
        print(f"\n❌ ERRORS ({len(all_errors)}):")
        for error in all_errors:
            print(f"  - {error}")

    if all_warnings:
        print(f"\n⚠️  WARNINGS ({len(all_warnings)}):")
        for warning in all_warnings:
            print(f"  - {warning}")

    if not all_errors and not all_warnings:
        print("\n✅ All checks passed!")

    return len(all_errors) == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_tools.py <tools.json>")
        sys.exit(1)

    path = sys.argv[1]
    success = validate_tools(path)
    sys.exit(0 if success else 1)
