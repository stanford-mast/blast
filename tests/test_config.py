"""Tests for configurations."""

import pytest
import yaml
from browser_use.browser import BrowserSession

from blastai import Engine
from blastai.config import Constraints, Settings
from blastai.resource_factory import _build_browser_args


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
        assert engine.settings.browser_cdp_url is None
    finally:
        await engine.stop()


def test_external_browser_cdp_url():
    """Accept both HTTP discovery URLs and direct WebSocket URLs."""
    assert Settings(browser_cdp_url="https://browser.example/cdp").browser_cdp_url == "https://browser.example/cdp"
    assert Settings(browser_cdp_url="wss://browser.example/cdp").browser_cdp_url == "wss://browser.example/cdp"


def test_external_browser_cdp_url_reaches_browser_session_args():
    settings = Settings(browser_cdp_url="wss://browser.example/cdp")
    constraints = Constraints(require_patchright=True, allowed_domains=["example.com"])

    browser_args = _build_browser_args("external-browser-test", constraints, settings)

    assert browser_args["cdp_url"] == "wss://browser.example/cdp"
    assert browser_args["allowed_domains"] == ["example.com"]
    assert browser_args["user_data_dir"] is None
    assert "executable_path" not in browser_args

    browser_session = BrowserSession(**browser_args)
    assert browser_session.cdp_url == "wss://browser.example/cdp"
    assert browser_session.is_local is False


@pytest.mark.parametrize("cdp_url", ["browser.example/cdp", "ftp://browser.example/cdp"])
def test_external_browser_cdp_url_rejects_unsupported_schemes(cdp_url):
    with pytest.raises(ValueError, match="browser_cdp_url must start"):
        Settings(browser_cdp_url=cdp_url)


def test_external_browser_cannot_also_launch_local_binary():
    with pytest.raises(ValueError, match="cannot be set together"):
        Settings(browser_cdp_url="wss://browser.example/cdp", local_browser_path="auto")


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
