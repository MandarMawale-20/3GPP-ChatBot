"""LLM provider abstraction.

Gemini is the MVP backend, but `LLMProvider` is the seam that lets a
different model be swapped in without touching the evidence gate,
verifier, or API layer.
"""

from __future__ import annotations

from typing import Protocol

from loguru import logger


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class GeminiProvider:
    """Production provider backed by Google's `google-genai` SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash") -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to use GeminiProvider")

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiProvider. Install it with `pip install google-genai`."
            ) from exc

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import types

        logger.info("Calling Gemini ({})", self._model_name)
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                # Low temperature: grounded QA, not creative generation —
                # reduces drift from the supplied evidence.
                temperature=0.1,
            ),
        )
        return response.text or ""


class FakeLLMProvider:
    """Deterministic provider for tests. Returns a canned response, or
    echoes back a configured answer — never calls a real API.
    """

    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


def build_default_llm_provider(api_key: str, model_name: str) -> LLMProvider:
    return GeminiProvider(api_key=api_key, model_name=model_name)
