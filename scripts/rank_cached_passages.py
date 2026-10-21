"""Rank cached abstracts using overlapping bi-encoder passages."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from verification_tools import sha256
from verify_claim_agent import dump, exclusive_run

MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


def windows(text, tokenizer, limit):
    offsets = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )["offset_mapping"]

    width = (
        limit
        - tokenizer.num_special_tokens_to_add(pair=False)
        - 8
    )

    if width < 16 or not offsets:
        raise ValueError(
            "Unsupported tokenizer or empty document"
        )

    output = []
    start = 0

    while start < len(offsets):
        stop = min(start + width, len(offsets))

        while True:
            left = (
                0 if start == 0
                else offsets[start][0]
            )
            right = (
                len(text)
                if stop == len(offsets)
                else offsets[stop - 1][1]
            )

            passage = text[left:right]

            count = len(
                tokenizer(
                    passage,
                    truncation=False,
                )["input_ids"]
            )

            if count <= limit:
                break

            if stop <= start + 1:
                raise ValueError(
                    "A source span cannot fit "
                    "without truncation"
                )

            stop -= max(1, (stop - start) // 8)

        output.append({
            "start_char": left,
            "end_char": right,
            "text": passage,
            "token_count": count,
        })

        if stop == len(offsets):
            break

        overlap = min(
            64,
            max(1, (stop - start) // 4),
        )
        start = max(start + 1, stop - overlap)

    return output


def rerank(data, encoder):
    claim = data["claim"]["claim_text"]
    tokenizer = encoder.tokenizer
    limit = encoder.max_seq_length

    claim_length = len(
        tokenizer(
            claim,
            truncation=False,
        )["input_ids"]
    )

    if claim_length > limit:
        raise ValueError(
            "Claim exceeds model limit; "
            "review scope before ranking"
        )

    baseline = {
        article["pmid"]: article["bi_encoder_score"]
        for article in data["wide_pool_ranking"]
    }

    articles = []
    texts = []

    for pmid, source in data["all_fetched_records"].items():
        if source["pmid"] != pmid:
            raise ValueError("Source PMID mismatch")

        if not source["text"].strip():
            continue

        document = (
            source["title"]
            + "\n\n"
            + source["text"]
        )

        spans = windows(document, tokenizer, limit)
        begin = len(texts)

        texts.extend(
            span["text"] for span in spans
        )

        articles.append({
            **source,
            "passages": spans,
            "_begin": begin,
            "baseline_bi_encoder_score": baseline.get(pmid),
        })

    if not texts:
        return []

    query = encoder.encode(
        [claim],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=4,
        show_progress_bar=False,
    )

    scores = (query @ embeddings.T).ravel()

    if len(scores) != len(texts):
        raise ValueError(
            "Embedding result count mismatch"
        )

    for article in articles:
        begin = article.pop("_begin")

        for index, span in enumerate(article["passages"]):
            score = float(scores[begin + index])

            if not -1.0001 <= score <= 1.0001:
                raise ValueError(
                    "Nonfinite or invalid cosine similarity"
                )

            span["score"] = score

        article["passage_score"] = max(
            passage["score"]
            for passage in article["passages"]
        )

        article["best_passage_index"] = max(
            range(len(article["passages"])),
            key=lambda index: (
                article["passages"][index]["score"]
            ),
        )

    articles.sort(
        key=lambda article: (
            -article["passage_score"],
            int(article["pmid"]),
        )
    )

    return articles


def show(folder, report):
    print("CLAIM:", report["claim"]["claim_id"])
    print("ARTICLES:", len(report["ranking"]))
    print(
        "PASSAGES:",
        sum(
            len(article["passages"])
            for article in report["ranking"]
        ),
    )

    print(
        "\nPASSAGE RANKING "
        "(provisional relevance only)"
    )

    for article in report["ranking"][:10]:
        print(
            article["pmid"],
            "|",
            round(article["passage_score"], 4),
            "|",
            article["title"],
        )

    print(
        "REPORT:",
        folder / "passage-ranking.json",
    )


def run(args):
    raw = Path(args.diagnostic).read_bytes()
    data = json.loads(raw)
    folder = Path(args.run_dir).resolve()

    if Path(args.diagnostic).resolve().is_relative_to(folder):
        raise ValueError(
            "Keep the diagnostic input outside "
            "this output directory"
        )

    binding = {
        "profile": "cached-passage-ranking-v1",
        "input_sha256": sha256(raw),
        "script_sha256": sha256(
            Path(__file__).read_bytes()
        ),
        "model": MODEL,
    }

    with exclusive_run(folder):
        path = folder / "passage-ranking.json"

        if path.exists():
            report = json.loads(path.read_bytes())

            if report["binding"] != binding:
                raise ValueError(
                    "Inputs/code changed; "
                    "use a new run directory"
                )

            show(folder, report)
            return

        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            MODEL,
            device="cpu",
        )

        ranking = rerank(data, encoder)

        config = getattr(
            getattr(encoder[0], "auto_model", None),
            "config",
            None,
        )

        report = {
            "binding": binding,
            "claim": data["claim"],
            "ranking": ranking,
            "max_seq_length": encoder.max_seq_length,
            "resolved_model_revision": getattr(
                config,
                "_commit_hash",
                None,
            ),
            "source_diagnostic": str(
                Path(args.diagnostic).resolve()
            ),
            "offsets_relative_to": (
                "title + two newline characters + abstract"
            ),
            "aggregation": (
                "maximum passage similarity; "
                "may favor longer documents"
            ),
            "scope": (
                "Same cached candidate pool; complete "
                "available title/abstract coverage. "
                "No new PubMed search, cross-encoder, "
                "suitability assessment or claim verdict. "
                "Every saved passage fits the "
                "bi-encoder limit."
            ),
        }

        dump(path, report)
        show(folder, report)


# Cached workflow ranking v1.
def _check_workflow_ranking(path, claim, records, input_path):
    raw = path.read_bytes()
    report = json.loads(raw)
    expected = {
        pmid for pmid, item in records.items()
        if item["text"].strip()
    }
    ids = [item["pmid"] for item in report["ranking"]]

    if (
        report["binding"]["input_sha256"]
        != sha256(input_path.read_bytes())
        or report["claim"]["claim_id"] != claim["claim_id"]
        or report["claim"]["claim_text"] != claim["claim_text"]
        or len(ids) != len(set(ids))
        or set(ids) != expected
    ):
        raise ValueError(
            "Ranking output has stale inputs, wrong identity "
            "or missing sources"
        )

    if any(
        item["text"] != records[item["pmid"]]["text"]
        for item in report["ranking"]
    ):
        raise ValueError("Ranking changed an original abstract")

    return sha256(raw)


def run_workflow(args):
    """Rank saved candidates through the existing model runners."""
    source = Path(args.workflow).resolve()
    folder = Path(args.run_dir).resolve()

    if source.is_relative_to(folder):
        raise ValueError(
            "Keep the workflow input outside the ranking directory"
        )

    raw_summary = source.read_bytes()
    workflow = json.loads(raw_summary)
    rows = workflow["claims"]
    ids = [row["claim_id"] for row in rows]

    if not rows or len(ids) != len(set(ids)):
        raise ValueError(
            "Expected a nonempty workflow with unique claim IDs"
        )

    results = []

    with exclusive_run(folder):
        for row in rows:
            cid = row["claim_id"]
            result = {
                "claim_id": cid,
                "workflow_status": row["status"],
            }

            if row["status"] != "awaiting_relevance_screening":
                result["status"] = "skipped"
            else:
                try:
                    ref = row["retrieval"]
                    evidence_path = Path(
                        ref["evidence_path"]
                    ).resolve()

                    if evidence_path.is_relative_to(folder):
                        raise ValueError(
                            "Keep original evidence outside ranking outputs"
                        )

                    raw = evidence_path.read_bytes()
                    if sha256(raw) != ref["evidence_sha256"]:
                        raise ValueError(
                            "Saved retrieval evidence was modified"
                        )

                    evidence = json.loads(raw)
                    if len(evidence["claims"]) != 1:
                        raise ValueError(
                            "Expected one claim per saved evidence file"
                        )

                    original = evidence["claims"][0]
                    if (
                        original["claim_id"] != cid
                        or original["claim_text"] != row["claim_text"]
                    ):
                        raise ValueError(
                            "Workflow and evidence claim identity differ"
                        )

                    records = {}
                    for article in original["evidence"]:
                        pmid = str(article["pmid"])
                        text = article["abstract"]

                        if not pmid.isdigit() or pmid in records:
                            raise ValueError(
                                "Invalid or duplicate PMID"
                            )
                        if sha256(text) != article["abstract_sha256"]:
                            raise ValueError(
                                "Abstract hash mismatch: " + pmid
                            )

                        records[pmid] = {
                            **article,
                            "pmid": pmid,
                            "text": text,
                        }

                    if not any(
                        item["text"].strip()
                        for item in records.values()
                    ):
                        raise ValueError(
                            "No nonempty abstracts available to rank"
                        )

                    claim = {
                        key: value
                        for key, value in original.items()
                        if key != "evidence"
                    }
                    adapter = {
                        "profile": "cached-workflow-candidates-v1",
                        "source_evidence_path": str(evidence_path),
                        "source_evidence_sha256": ref["evidence_sha256"],
                        "claim": claim,
                        "all_fetched_records": records,
                        "wide_pool_ranking": [],
                    }

                    target = folder / sha256(cid)
                    adapter_path = target / "cached-candidates.json"

                    if adapter_path.exists():
                        if json.loads(adapter_path.read_bytes()) != adapter:
                            raise ValueError(
                                "Saved ranking input differs; "
                                "use a new ranking directory"
                            )
                    else:
                        dump(adapter_path, adapter)

                    print("\nPASSAGE RANKING:", cid, flush=True)
                    subprocess.run([
                        sys.executable,
                        "-B",
                        str(Path(__file__).resolve()),
                        "--diagnostic", str(adapter_path),
                        "--run-dir", str(target / "bi-encoder"),
                    ], check=True)

                    passage_path = (
                        target / "bi-encoder/passage-ranking.json"
                    )
                    passage_hash = _check_workflow_ranking(
                        passage_path, claim, records, adapter_path
                    )

                    result.update(
                        status="passage_ranked",
                        passage_report=str(passage_path),
                        passage_report_sha256=passage_hash,
                        source_evidence_sha256=ref["evidence_sha256"],
                        usable_abstracts=sum(
                            bool(item["text"].strip())
                            for item in records.values()
                        ),
                    )

                    if args.cross_encoder:
                        print("\nCROSS-ENCODER:", cid, flush=True)
                        subprocess.run([
                            sys.executable,
                            "-B",
                            str(
                                Path(__file__).with_name(
                                    "rerank_cross_encoder.py"
                                ).resolve()
                            ),
                            "--passage-report", str(passage_path),
                            "--run-dir", str(target / "cross-encoder"),
                        ], check=True)

                        cross_path = (
                            target
                            / "cross-encoder/cross-encoder-ranking.json"
                        )
                        cross_hash = _check_workflow_ranking(
                            cross_path, claim, records, passage_path
                        )
                        result.update(
                            status="passage_and_cross_encoder_ranked",
                            cross_encoder_report=str(cross_path),
                            cross_encoder_report_sha256=cross_hash,
                        )

                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    subprocess.CalledProcessError,
                ) as error:
                    result.update(
                        status="ranking_paused",
                        error=f"{type(error).__name__}: {error}",
                    )

            results.append(result)
            dump(folder / "ranking-summary.json", {
                "workflow_path": str(source),
                "workflow_sha256": sha256(raw_summary),
                "claims": results,
                "cross_encoder_requested": args.cross_encoder,
                "scope": (
                    "Cached source relevance ranking only; "
                    "no scientific verdict or review approval."
                ),
            })

            print(cid, "|", result["status"], flush=True)
            if result.get("error"):
                print("  ERROR:", result["error"], flush=True)

    print("SUMMARY:", folder / "ranking-summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--diagnostic",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
    )
    parser.add_argument(
        "--workflow", help="Saved workflow-summary.json"
    )
    parser.add_argument(
        "--cross-encoder", action="store_true"
    )

    args = parser.parse_args()

    if bool(args.diagnostic) == bool(args.workflow):
        parser.error(
            "Supply exactly one of --diagnostic or --workflow"
        )

    if args.workflow:
        run_workflow(args)
    else:
        if args.cross_encoder:
            parser.error(
                "--cross-encoder is available with --workflow"
            )
        run(args)