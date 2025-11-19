"""Tests for configurations."""

import pytest
import yaml

from blastai import Engine


@pytest.mark.asyncio
async def test_default_config():
    """Test loading default configuration."""
    engine = await Engine.create()
    try:
        # Check default values from default_config.yaml
        # allow_parallelism:
        #   first_of_n: false
        assert engine.constraints.allow_parallelism["first_of_n"] is False
        assert engine.constraints.first_of_n_num_copies == 3
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_custom_config(tmp_path):
    """Test loading custom configuration."""
    config_content = {
        "constraints": {
            "allow_parallelism": {"task": True, "data": True, "first_of_n": True},
            "first_of_n_num_copies": 5,
        }
    }

    config_file = tmp_path / "custom_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    engine = await Engine.create(config_path=str(config_file))
    try:
        assert engine.constraints.allow_parallelism["first_of_n"] is True
        assert engine.constraints.first_of_n_num_copies == 5
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_partial_override(tmp_path):
    """Test partially overriding configuration."""
    # Only override first_of_n_num_copies, keep others default
    config_content = {"constraints": {"first_of_n_num_copies": 10}}

    config_file = tmp_path / "partial_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    engine = await Engine.create(config_path=str(config_file))
    try:
        # allow_parallelism should still be default (first_of_n: False)
        assert engine.constraints.allow_parallelism["first_of_n"] is False
        assert engine.constraints.first_of_n_num_copies == 10
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_partial_dict_override(tmp_path):
    """Test partially overriding a dictionary field (allow_parallelism)."""
    config_content = {"constraints": {"allow_parallelism": {"first_of_n": True}}}

    config_file = tmp_path / "partial_dict_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    engine = await Engine.create(config_path=str(config_file))
    try:
        assert engine.constraints.allow_parallelism["first_of_n"] is True

        # "task" and "data" keys should be preserved from default config.
        assert engine.constraints.allow_parallelism["task"] is True
        assert engine.constraints.allow_parallelism["data"] is False

    finally:
        await engine.stop()
