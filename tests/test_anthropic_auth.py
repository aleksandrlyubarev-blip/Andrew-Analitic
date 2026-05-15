"""
Tests for core/anthropic_auth.py — auth-mode toggle parsing and guard.

Runs fully offline — no network, no LiteLLM imports.
"""

import logging

import pytest

from core import anthropic_auth


def test_default_mode_when_env_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_MODE", raising=False)
    assert anthropic_auth.get_auth_mode() == "api_key"
    assert anthropic_auth.is_sdk_mode() is False


def test_explicit_api_key_mode(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "api_key")
    assert anthropic_auth.get_auth_mode() == "api_key"


def test_sdk_mode(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "sdk")
    assert anthropic_auth.get_auth_mode() == "sdk"
    assert anthropic_auth.is_sdk_mode() is True


@pytest.mark.parametrize("raw", ["SDK", "  sdk  ", "Sdk"])
def test_mode_is_case_and_whitespace_insensitive(monkeypatch, raw):
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", raw)
    assert anthropic_auth.get_auth_mode() == "sdk"


def test_unknown_mode_falls_back_to_default(monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "magic")
    with caplog.at_level(logging.WARNING, logger="anthropic_auth"):
        assert anthropic_auth.get_auth_mode() == "api_key"
    assert any("Unknown ANTHROPIC_AUTH_MODE" in r.message for r in caplog.records)


def test_empty_mode_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "")
    assert anthropic_auth.get_auth_mode() == "api_key"


def test_require_api_key_mode_passes_in_default(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_MODE", raising=False)
    anthropic_auth.require_api_key_mode("test_component")


def test_require_api_key_mode_raises_in_sdk(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "sdk")
    with pytest.raises(NotImplementedError, match="test_component"):
        anthropic_auth.require_api_key_mode("test_component")
