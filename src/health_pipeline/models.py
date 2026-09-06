"""Pydantic contracts shared by pipeline stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TranscriptSegment(BaseModel):
    """A piece of spoken or on-screen text with optional video timing."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)


class Transcript(BaseModel):
    """Unified ASR and OCR output passed to claim extraction."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    timestamps: list[TranscriptSegment] = Field(default_factory=list)
    on_screen_text: list[TranscriptSegment] = Field(default_factory=list)


class AtomicClaim(BaseModel):
    """One independently verifiable factual statement."""

    model_config = ConfigDict(extra="forbid")

    claim_text: str = Field(min_length=1)
    source_span: TranscriptSegment
    is_factual: bool


class ClaimExtractionResult(BaseModel):
    """Structured response emitted by a claim extraction agent."""

    model_config = ConfigDict(extra="forbid")

    claims: list[AtomicClaim] = Field(default_factory=list)


class Evidence(BaseModel):
    """A retrieved passage and its provenance."""

    model_config = ConfigDict(extra="forbid")

    passage: str = Field(min_length=1)
    source_url: str
    title: str = ""
    source_domain: str = ""
    relevance_score: float | None = None


class Verdict(BaseModel):
    """Verification result for one claim."""

    model_config = ConfigDict(extra="forbid")

    label: Literal[
        "Supported", "Refuted", "Misleading", "InsufficientEvidence"
    ]
    confidence: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=1)
