"""Compare PubMed candidate budgets with fixed queries and one bi-encoder."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import requests
from verification_tools import parse_pubmed, sha256
from verify_claim_agent import dump, exclusive_run

MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


def validate_response(endpoint, raw):
    if endpoint == "esearch.fcgi":
        data = json.loads(raw)
        result = data.get("esearchresult")

        if (
            data.get("error")
            or not isinstance(result, dict)
            or result.get("ERROR")
        ):
            raise ValueError("PubMed returned a search error")

        ids = result.get("idlist")

        if not isinstance(ids, list) or any(
            not isinstance(p, str) or not p.isdigit()
            for p in ids
        ):
            raise ValueError("Invalid PubMed PMID list")

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate PMIDs in one search response"
            )
    else:
        root = ET.fromstring(raw)

        if (
            root.tag != "PubmedArticleSet"
            or root.find(".//ERROR") is not None
        ):
            raise ValueError(
                "Unexpected PubMed XML or service error"
            )


class NCBI:
    def __init__(self, folder, state, email):
        self.folder = folder
        self.state = state
        self.email = email
        self.last_request = 0.0

    def get(self, endpoint, params):
        key = sha256(
            json.dumps([endpoint, params], sort_keys=True)
        )
        cached = self.state["cache"].get(key)

        if cached:
            raw = (
                self.folder / cached["file"]
            ).read_bytes()

            if sha256(raw) != cached["sha256"]:
                raise ValueError(
                    "Cached PubMed response changed"
                )

            validate_response(endpoint, raw)
            return raw, cached["retrieved_at"]

        used = sum(
            event["key"] == key
            for event in self.state["events"]
        )

        for attempt in range(used, 2):
            event = {
                "key": key,
                "endpoint": endpoint,
                "status": "started",
            }

            self.state["events"].append(event)
            dump(self.folder / "state.json", self.state)

            try:
                delay = max(
                    0,
                    0.4 - (
                        time.monotonic() - self.last_request
                    ),
                )
                time.sleep(delay)
                self.last_request = time.monotonic()

                response = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/"
                    "entrez/eutils/" + endpoint,
                    params={
                        **params,
                        "email": self.email,
                        "tool": "health_retrieval_diagnostic",
                    },
                    timeout=(10, 60),
                )
                response.raise_for_status()

                raw = response.content
                validate_response(endpoint, raw)

                suffix = (
                    "json"
                    if endpoint == "esearch.fcgi"
                    else "xml"
                )
                filename = (
                    "pubmed-cache/" + key + "." + suffix
                )
                target = self.folder / filename
                target.parent.mkdir(
                    parents=True, exist_ok=True
                )
                target.write_bytes(raw)

                stamp = datetime.now(
                    timezone.utc
                ).isoformat()

                self.state["cache"][key] = {
                    "file": filename,
                    "sha256": sha256(raw),
                    "retrieved_at": stamp,
                }
                event["status"] = "cached"

                return raw, stamp

            except (
                requests.RequestException,
                ValueError,
                ET.ParseError,
            ) as error:
                event.update(
                    status="error",
                    error=str(error),
                )

                code = getattr(
                    getattr(error, "response", None),
                    "status_code",
                    None,
                )

                if attempt == 1 or (
                    code is not None and code < 500
                ):
                    raise

                time.sleep(1)

            finally:
                dump(
                    self.folder / "state.json",
                    self.state,
                )

        raise ValueError(
            "PubMed request budget exhausted; "
            "inspect state.json"
        )


def collect(api, queries):
    searches = []
    small = set()
    all_ids = set()

    for query in queries:
        raw, stamp = api.get(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": 50,
                "sort": "relevance",
            },
        )

        result = json.loads(raw)["esearchresult"]
        ids = result["idlist"]

        if len(ids) > 50:
            raise ValueError(
                "PubMed returned more IDs than requested"
            )

        small.update(ids[:5])
        all_ids.update(ids)

        searches.append({
            "query": query,
            "total_matches": result.get("count"),
            "query_translation": result.get(
                "querytranslation"
            ),
            "warnings": result.get("warninglist", {}),
            "returned_pmids": ids,
            "retrieved_at": stamp,
        })

    records = {}
    ordered = sorted(all_ids, key=int)

    for start in range(0, len(ordered), 50):
        batch = ordered[start:start + 50]

        raw, stamp = api.get(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            },
        )

        for article in parse_pubmed(raw):
            pmid = article["pmid"]

            if pmid not in batch or pmid in records:
                raise ValueError(
                    "Unexpected or duplicate fetched PMID"
                )

            article["retrieved_at"] = stamp
            article["matched_queries"] = [
                i + 1
                for i, search in enumerate(searches)
                if pmid in search["returned_pmids"]
            ]

            records[pmid] = article

    return searches, small, all_ids, records


def rank(claim_text, records, encoder=None):
    articles = [
        dict(records[pmid])
        for pmid in sorted(records, key=int)
        if records[pmid]["text"].strip()
    ]

    if not articles:
        return [], {
            "model": MODEL,
            "status": "no_abstracts_to_rank",
        }

    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            MODEL, device="cpu"
        )

    texts = [
        article["title"] + "\n" + article["text"]
        for article in articles
    ]

    lengths = [
        len(ids)
        for ids in encoder.tokenizer(
            texts, truncation=False
        )["input_ids"]
    ]

    query_length = len(
        encoder.tokenizer(
            [claim_text], truncation=False
        )["input_ids"][0]
    )

    query = encoder.encode(
        [claim_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    docs = encoder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=4,
        show_progress_bar=False,
    )

    scores = (query @ docs.T).ravel()

    for article, score, length in zip(
        articles, scores, lengths
    ):
        article.update(
            bi_encoder_score=float(score),
            token_count=length,
            embedding_truncated=(
                length > encoder.max_seq_length
            ),
        )

    articles.sort(
        key=lambda article: (
            -article["bi_encoder_score"],
            int(article["pmid"]),
        )
    )

    return articles, {
        "model": MODEL,
        "max_seq_length": encoder.max_seq_length,
        "query_token_count": query_length,
        "query_truncated": (
            query_length > encoder.max_seq_length
        ),
        "potentially_truncated_abstracts": sum(
            article["embedding_truncated"]
            for article in articles
        ),
        "note": (
            "Full abstracts retained for review; "
            "model input may be truncated. "
            "Scores are not truth probabilities."
        ),
    }


def write_review(folder, report):
    lines = [
        "CLAIM: " + report["claim"]["claim_text"],
        "ASSISTED diagnostic selection. "
        "No claim verdict or retrieval-quality grade.",
        *report["review_questions"],
    ]

    for name in (
        "small_pool_ranking",
        "wide_pool_ranking",
    ):
        lines.append("\n" + name.upper())

        for number, article in enumerate(
            report[name][:10], 1
        ):
            lines.extend([
                f"\n{number}. PMID {article['pmid']} "
                f"| {article['title']}",
                article["url"],
                "Publication types: "
                + json.dumps(article["publication_types"]),
                "Correction/retraction links: "
                + json.dumps(article["correction_notices"]),
                "Full available abstract:\n"
                + article["text"],
                "REVIEW: relevant / partial / "
                "unrelated / unclear = ____",
                "EXPOSURE + COMPARATOR: ____ "
                "| ACTUAL OUTCOME: ____ | DURATION: ____",
                "RESULT + LIMITATIONS: ____",
            ])

    # Preserve any notes added during manual review.
    path = folder / "review.txt"
    if not path.exists():
        path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )


def summary(folder, report):
    print("CLAIM:", report["claim"]["claim_id"])
    print(
        "SMALL POOL PMIDS:",
        report["counts"]["small_pool_pmids"],
    )
    print(
        "WIDE POOL PMIDS:",
        report["counts"]["wide_pool_pmids"],
    )
    print(
        "WITH ABSTRACTS:",
        len(report["wide_pool_ranking"]),
    )
    print(
        "UNRESOLVED PMIDS:",
        report["unresolved_pmids"],
    )
    print(
        "WITHOUT ABSTRACTS:",
        report["empty_abstract_pmids"],
    )
    print(
        "EMBEDDING INPUT CHECK:",
        report["encoder"],
    )

    for name in (
        "small_pool_ranking",
        "wide_pool_ranking",
    ):
        print("\n" + name.upper())

        for article in report[name][:10]:
            print(
                article["pmid"],
                "|",
                round(article["bi_encoder_score"], 4),
                "|",
                article["title"],
            )

    print("\nREPORT:", folder / "diagnostic.json")
    print("REVIEW:", folder / "review.txt")


def run(args):
    plan_raw = Path(args.plan).read_bytes()
    plan = json.loads(plan_raw)
    claim, queries = plan["claim"], plan["queries"]

    if (
        not claim.get("claim_id")
        or not claim.get("claim_text")
    ):
        raise ValueError(
            "Plan needs a claim ID and wording"
        )

    if (
        not isinstance(queries, list)
        or not 1 <= len(queries) <= 3
        or any(
            not isinstance(query, str)
            or not query.strip()
            for query in queries
        )
    ):
        raise ValueError(
            "Supply 1-3 nonempty PubMed queries"
        )

    transcript_raw = Path(
        plan["transcript_file"]
    ).read_bytes()
    transcript = json.loads(transcript_raw)
    quote = plan.get("source_quote")

    if (
        not isinstance(quote, str)
        or not quote.strip()
        or quote not in transcript["text"]
    ):
        raise ValueError(
            "Source quote is not present "
            "in the transcript"
        )

    if not args.email:
        raise ValueError(
            "Set NCBI_EMAIL to your contact email"
        )

    folder = Path(args.run_dir).resolve()

    if any(
        path.resolve().is_relative_to(folder)
        for path in (
            Path(args.plan),
            Path(plan["transcript_file"]),
        )
    ):
        raise ValueError(
            "Keep input files outside "
            "the output directory"
        )

    import verification_tools

    binding = {
        "profile": "retrieval-budget-diagnostic-v1",
        "plan_sha256": sha256(plan_raw),
        "transcript_sha256": sha256(transcript_raw),
        "script_sha256": sha256(
            Path(__file__).read_bytes()
        ),
        "parser_sha256": sha256(
            Path(verification_tools.__file__).read_bytes()
        ),
        "model": MODEL,
    }

    with exclusive_run(folder):
        path = folder / "state.json"

        state = (
            json.loads(path.read_bytes())
            if path.exists()
            else {
                "binding": binding,
                "cache": {},
                "events": [],
            }
        )

        if state["binding"] != binding:
            raise ValueError(
                "Inputs/code changed; "
                "use a new run directory"
            )

        for event in state["events"]:
            if event["status"] == "started":
                event["status"] = "interrupted_unknown"

        dump(path, state)
        dump(folder / "plan.json", plan)
        dump(
            folder / "transcript-snapshot.json",
            transcript,
        )

        searches, small, all_ids, records = collect(
            NCBI(folder, state, args.email),
            queries,
        )

        ranked, metadata = rank(
            claim["claim_text"], records
        )

        report = {
            "binding": binding,
            "claim": claim,
            "searches": searches,
            "review_questions": plan.get(
                "review_questions",
                [
                    "Does the study address the exact "
                    "exposure, outcome and context?",
                    "Record comparator, population, "
                    "duration, results and limitations.",
                ],
            ),
            "counts": {
                "small_pool_pmids": len(small),
                "wide_pool_pmids": len(all_ids),
            },
            "unresolved_pmids": sorted(
                all_ids - set(records),
                key=int,
            ),
            "empty_abstract_pmids": sorted(
                (
                    pmid
                    for pmid, article in records.items()
                    if not article["text"].strip()
                ),
                key=int,
            ),
            "all_fetched_records": records,
            "encoder": metadata,
            "small_pool_ranking": [
                article for article in ranked
                if article["pmid"] in small
            ],
            "wide_pool_ranking": ranked,
            "scope": (
                "Assisted queries; compares 5 versus 50 "
                "results per query with the same bi-encoder. "
                "No cross-encoder, scientific verification "
                "or adjudication. Not an evaluation of "
                "automatic claim extraction or query planning."
            ),
        }

        dump(folder / "diagnostic.json", report)
        write_review(folder, report)
        summary(folder, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--email",
        default=os.environ.get("NCBI_EMAIL"),
    )
    run(parser.parse_args())