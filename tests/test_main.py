"""Tests for ai_essentials.main module."""

from ai_essentials.main import greet, main


def test_greet_returns_greeting():
    result = greet("Alice")
    assert result == "Hello, Alice! Welcome to AI Essentials."


def test_greet_with_world():
    result = greet("World")
    assert result == "Hello, World! Welcome to AI Essentials."


def test_main_runs_without_error(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello, World! Welcome to AI Essentials." in captured.out
