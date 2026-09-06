from health_pipeline.claim_extraction import ClaimExtractor
from health_pipeline.models import (
    AtomicClaim,
    ClaimExtractionResult,
    Transcript,
    TranscriptSegment,
)


class FakeLLM:
    def __init__(self, result: ClaimExtractionResult) -> None:
        self.result = result
        self.prompt = ""
        self.response_model = None

    def generate_structured(self, prompt: str, response_model):
        self.prompt = prompt
        self.response_model = response_model
        return self.result


def test_extractor_returns_typed_atomic_claims() -> None:
    span = TranscriptSegment(text="Vaccines reduce severe disease.", start=2, end=4)
    expected = ClaimExtractionResult(
        claims=[
            AtomicClaim(
                claim_text="Vaccines reduce severe disease.",
                source_span=span,
                is_factual=True,
            )
        ]
    )
    llm = FakeLLM(expected)

    result = ClaimExtractor(llm).extract(
        Transcript(text="Vaccines reduce severe disease.")
    )

    assert result == expected
    assert llm.response_model is ClaimExtractionResult
    assert "Exclude opinions" in llm.prompt


def test_extractor_preserves_non_factual_classification() -> None:
    result = ClaimExtractionResult(
        claims=[
            AtomicClaim(
                claim_text="This is the best supplement.",
                source_span=TranscriptSegment(text="This is the best supplement."),
                is_factual=False,
            )
        ]
    )
    assert ClaimExtractor(FakeLLM(result)).extract(Transcript(text="opinion")) == result
