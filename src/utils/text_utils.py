"""Text formatting helpers for pipeline outputs."""

from __future__ import annotations


def format_rag_output(food_name: str, description: str) -> str:
    """Combine food name and description for display."""
    return f"{food_name}: {description}"
