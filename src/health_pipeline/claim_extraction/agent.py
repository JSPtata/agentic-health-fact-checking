"""LLM-backed extraction of atomic factual claims."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from ..models import ClaimExtractionResult, Transcript

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class LLMClient(Protocol):
    """Minimal structured-generation contract for local or hosted LLMs."""

    def generate_structured(
        self, prompt: str, response_model: type[ResponseT]
    ) -> ResponseT: ...


class ClaimExtractor:
    """Extracts claims while keeping model-specific prompting out of the pipeline."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def extract(self, transcript: Transcript) -> ClaimExtractionResult:
        prompt = self._build_prompt(transcript)
        result = self.llm_client.generate_structured(prompt, ClaimExtractionResult)
        if not isinstance(result, ClaimExtractionResult):
            raise TypeError(
                "LLM client returned an unexpected type; expected "
                "ClaimExtractionResult."
            )
        return result

    @staticmethod
    def _build_prompt(transcript: Transcript) -> str:
        return (
            "Extract only independently verifiable health claims from the transcript. "
            "Exclude opinions, advertisements, calls to action, greetings, and "
            "questions. Split compound statements into atomic claims. Set "
            "is_factual=false for anything that is not a factual assertion. "
            "For every claim, include the exact supporting source_span text and "
            "timestamps when available. Return only the requested structured schema.\n\n"
            f"Transcript:\n{transcript.model_dump_json(indent=2)}"
        )
