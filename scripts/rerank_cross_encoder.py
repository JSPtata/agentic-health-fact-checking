"""Biomedical cross-encoder comparison on cached articles."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path

import rank_cached_passages
from rank_cached_passages import windows
from verification_tools import sha256
from verify_claim_agent import dump, exclusive_run

MODEL = "ncbi/MedCPT-Cross-Encoder"


def load_input(raw):
    data = json.loads(raw)
    claim = data["claim"]

    if (
        not isinstance(claim.get("claim_text"), str)
        or not claim["claim_text"].strip()
    ):
        raise ValueError("Missing claim wording")

    seen = set()

    for source in data["ranking"]:
        pmid = source["pmid"]

        if (
            not isinstance(pmid, str)
            or not pmid.isdigit()
            or pmid in seen
        ):
            raise ValueError(
                "Missing or duplicate source identity"
            )

        if (
            not isinstance(source.get("text"), str)
            or not source["text"].strip()
        ):
            raise ValueError("Source has no abstract")

        seen.add(pmid)

    return data


def prepare(query, source, tokenizer, limit):
    query_tokens = len(
        tokenizer(
            query,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
    )

    single = tokenizer.num_special_tokens_to_add(
        pair=False
    )
    paired = tokenizer.num_special_tokens_to_add(
        pair=True
    )

    budget = limit - query_tokens - paired + single

    if budget < 32:
        raise ValueError(
            "Claim leaves too little room for evidence; "
            "review claim scope"
        )

    document = (
        source["title"]
        + "\n\n"
        + source["text"]
    )

    spans = windows(document, tokenizer, budget)

    for span in spans:
        length = len(
            tokenizer(
                query,
                span["text"],
                truncation=False,
                verbose=False,
            )["input_ids"]
        )

        if length > limit:
            raise ValueError(
                "Claim/passage pair exceeds model limit; "
                "no text truncated"
            )

        span["pair_token_count"] = length

    if (
        spans[0]["start_char"] != 0
        or spans[-1]["end_char"] != len(document)
        or any(
            right["start_char"] > left["end_char"]
            for left, right in zip(spans, spans[1:])
        )
    ):
        raise ValueError(
            "Passages do not cover the complete source"
        )

    return spans


def predict(
    query,
    spans,
    tokenizer,
    model,
    limit,
    batch_size=2,
):
    import torch

    scores = []

    with torch.inference_mode():
        for start in range(0, len(spans), batch_size):
            batch = spans[start:start + batch_size]

            encoded = tokenizer(
                [query] * len(batch),
                [passage["text"] for passage in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
                verbose=False,
            )

            if encoded["input_ids"].shape[1] > limit:
                raise ValueError(
                    "Encoded batch exceeds model limit"
                )

            logits = model(**encoded).logits

            if tuple(logits.shape) != (len(batch), 1):
                raise ValueError(
                    "Expected one relevance logit per "
                    "claim/passage pair"
                )

            values = (
                logits[:, 0]
                .detach()
                .cpu()
                .tolist()
            )

            if any(
                not math.isfinite(value)
                for value in values
            ):
                raise ValueError(
                    "Nonfinite cross-encoder score"
                )

            scores.extend(values)

    return scores


def score_article(
    query,
    source,
    tokenizer,
    model,
    limit,
):
    spans = prepare(
        query, source, tokenizer, limit
    )

    scores = predict(
        query, spans, tokenizer, model, limit
    )

    if len(scores) != len(spans):
        raise ValueError(
            "Passage/score count mismatch"
        )

    for span, score in zip(spans, scores):
        span["score"] = score

    card = {
        key: value
        for key, value in source.items()
        if key not in (
            "passages",
            "passage_score",
            "best_passage_index",
        )
    }

    card.update(
        bi_encoder_passage_score=source["passage_score"],
        cross_encoder_score=max(scores),
        cross_encoder_passages=spans,
        best_cross_encoder_passage_index=max(
            range(len(scores)),
            key=scores.__getitem__,
        ),
    )

    return card


def show(folder, report):
    print("CLAIM:", report["claim"]["claim_id"])
    print("ARTICLES:", len(report["ranking"]))

    previous = {
        pmid: index + 1
        for index, pmid in enumerate(
            report["bi_encoder_order"]
        )
    }

    print(
        "\nCROSS-ENCODER RANKING "
        "(relevance scores, not claim verdicts)"
    )

    for index, article in enumerate(
        report["ranking"][:10], 1
    ):
        print(
            index,
            "|",
            article["pmid"],
            "| CE:",
            round(article["cross_encoder_score"], 4),
            "| previous rank:",
            previous[article["pmid"]],
            "|",
            article["title"],
        )

    print(
        "REPORT:",
        folder / "cross-encoder-ranking.json",
    )


def run(args):
    raw = Path(args.passage_report).read_bytes()
    data = load_input(raw)
    folder = Path(args.run_dir).resolve()

    if Path(args.passage_report).resolve().is_relative_to(folder):
        raise ValueError(
            "Keep the input report outside "
            "this output directory"
        )

    binding = {
        "profile": "biomedical-cross-encoder-v1",
        "input_sha256": sha256(raw),
        "script_sha256": sha256(
            Path(__file__).read_bytes()
        ),
        "model": MODEL,
        "chunker_sha256": sha256(
            Path(rank_cached_passages.__file__).read_bytes()
        ),
        "versions": {
            package: version(package)
            for package in ("torch", "transformers")
        },
    }

    with exclusive_run(folder):
        result_path = (
            folder / "cross-encoder-ranking.json"
        )

        if result_path.exists():
            report = json.loads(
                result_path.read_bytes()
            )

            if report["binding"] != binding:
                raise ValueError(
                    "Inputs/code changed; "
                    "use a new run directory"
                )

            show(folder, report)
            return

        path = folder / "state.json"

        state = (
            json.loads(path.read_bytes())
            if path.exists()
            else {
                "binding": binding,
                "model_identity": None,
                "outputs": {},
            }
        )

        if state["binding"] != binding:
            raise ValueError(
                "Inputs/code changed; "
                "use a new run directory"
            )

        dump(path, state)

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        print(
            "Loading biomedical cross-encoder:",
            MODEL,
            flush=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            use_fast=True,
        )

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(MODEL)
            .to("cpu")
            .eval()
        )

        limit = min(
            512,
            model.config.max_position_embeddings,
        )

        identity = {
            "revision": getattr(
                model.config,
                "_commit_hash",
                None,
            ),
            "limit": limit,
        }

        if state["model_identity"] not in (
            None,
            identity,
        ):
            raise ValueError(
                "Model revision changed; "
                "use a new run directory"
            )

        state["model_identity"] = identity
        dump(path, state)

        for index, source in enumerate(
            data["ranking"], 1
        ):
            pmid = source["pmid"]

            if pmid not in state["outputs"]:
                try:
                    card = score_article(
                        data["claim"]["claim_text"],
                        source,
                        tokenizer,
                        model,
                        limit,
                    )

                    state["outputs"][pmid] = card
                    state.pop("last_error", None)

                except (
                    ValueError,
                    RuntimeError,
                    OSError,
                ) as error:
                    state["last_error"] = {
                        "pmid": pmid,
                        "error": str(error),
                    }
                    raise

                finally:
                    dump(path, state)

            if (
                index == 1
                or index % 10 == 0
                or index == len(data["ranking"])
            ):
                print(
                    f"Scored {index}/"
                    f"{len(data['ranking'])} articles",
                    flush=True,
                )

        ranked = [
            state["outputs"][source["pmid"]]
            for source in data["ranking"]
        ]

        ranked.sort(
            key=lambda article: (
                -article["cross_encoder_score"],
                int(article["pmid"]),
            )
        )

        report = {
            "binding": binding,
            "model_identity": identity,
            "claim": data["claim"],
            "ranking": ranked,
            "bi_encoder_order": [
                source["pmid"]
                for source in data["ranking"]
            ],
            "offsets_relative_to": (
                "title + two newline characters + abstract"
            ),
            "aggregation": (
                "maximum passage relevance logit; "
                "may favor longer sources"
            ),
            "scope": (
                "Same cached articles and claim; passages "
                "use the cross-encoder tokenizer and pair "
                "limit. Biomedical retrieval model, not a "
                "scientific verifier. Scores are not "
                "probabilities and are not comparable with "
                "cosine similarities. No automatic "
                "exclusions or claim verdicts."
            ),
        }

        dump(result_path, report)
        show(folder, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--passage-report",
        required=True,
    )
    parser.add_argument(
        "--run-dir",
        required=True,
    )
    run(parser.parse_args())