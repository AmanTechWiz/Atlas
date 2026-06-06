"""Helpers to translate raw Gemini API exceptions into user-friendly messages.

Used by the workflow nodes and the Streamlit UI so users see a clear
"Gemini API quota exhausted" banner instead of a 429 stack trace
disguised as a "low confidence" warning.
"""

from __future__ import annotations


def is_quota_error(exc: Exception) -> bool:
    """True if the exception is a Gemini 429 / RESOURCE_EXHAUSTED / quota issue."""
    msg = str(exc).lower()
    return (
        "resource_exhausted" in msg
        or "quota exceeded" in msg
        or "rate limit" in msg
        or "quota_metric" in msg
        or "429" in msg
    )


def is_model_not_found(exc: Exception) -> bool:
    """True if the exception indicates a 404 / unknown model name."""
    msg = str(exc).lower()
    return "404" in msg or "not found" in msg or "model_not_found" in msg


def friendly_api_error(exc: Exception) -> str:
    """Convert a raw Gemini exception into a short, actionable user message."""
    if is_quota_error(exc):
        return (
            "Gemini API quota exhausted. The free tier allows 20 requests/day on this "
            "model; the quota resets at midnight UTC. Please try again later, switch to a "
            "paid plan, or set GEMINI_MODEL in `.env` to a model with available quota."
        )
    if is_model_not_found(exc):
        return (
            f"Gemini model not available ({exc}). Check the `GEMINI_MODEL` value in `.env` "
            f"and confirm it is on the supported list for this API key."
        )
    return f"Gemini API call failed: {exc}"
