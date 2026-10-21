"""Refresh PubMed candidates for pending claims; no model calls."""
import argparse
import json
import os
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import requests

from adjudicate_batch import sha
from verification_tools import parse_pubmed
from verify_claim_agent import dump, exclusive_run


def clean(text):
    return " ".join(str(text or "").split())


def load_jobs(worklist, evidence):
    rows = json.loads(worklist)["claims"]
    candidates = {}

    for item in json.loads(evidence)["claims"]:
        cid = item["claim_id"]

        if cid in candidates:
            raise ValueError(
                f"Duplicate claim in evidence: {cid}"
            )

        candidates[cid] = item

    jobs = []
    seen = set()

    for row in rows:
        cid = row["claim_id"]

        if (
            cid in seen
            or not re.fullmatch(r"[A-Za-z0-9_-]+", cid)
        ):
            raise ValueError(
                f"Duplicate or unsafe claim ID: {cid}"
            )

        seen.add(cid)

        if row["status"] != "verification_pending":
            continue

        claim = row["claim"]

        if (
            claim.get("claim_id") != cid
            or not clean(claim.get("claim_text"))
        ):
            raise ValueError(
                f"Claim identity mismatch: {cid}"
            )

        item = candidates.get(cid)

        if item is None:
            jobs.append({
                "claim": claim,
                "pmids": [],
                "input_issue": "No evidence record supplied",
            })
            continue

        if clean(item["claim_text"]) != clean(claim["claim_text"]):
            raise ValueError(
                f"Evidence claim wording mismatch: {cid}"
            )

        pmids = [
            str(article["pmid"])
            for article in item.get("evidence", [])
        ]

        if any(
            not re.fullmatch(r"[1-9][0-9]*", pmid)
            for pmid in pmids
        ):
            raise ValueError(f"Invalid PMID: {cid}")

        jobs.append({
            "claim": claim,
            "pmids": sorted(set(pmids)),
            "input_issue": None,
        })

    return jobs


def fetch_batch(folder, state, ids, email):
    key = sha(",".join(ids).encode())[:20]
    cache = state["cache"].get(key)

    if cache:
        raw = (folder / cache["file"]).read_bytes()

        if sha(raw) != cache["sha256"]:
            raise ValueError(
                "Cached PubMed response changed"
            )

        records = parse_pubmed(raw)

        for record in records:
            record["retrieved_at"] = cache["retrieved_at"][
                record["pmid"]
            ]

        return records

    used = sum(
        event["key"] == key
        for event in state["events"]
    )

    for attempt in range(used, 2):
        event = {
            "key": key,
            "pmids": ids,
            "status": "started",
        }

        state["events"].append(event)
        dump(folder / "preparation-state.json", state)

        print(
            f"Fetching {len(ids)} PMIDs; "
            f"attempt {attempt + 1}/2",
            flush=True,
        )

        try:
            response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/"
                "entrez/eutils/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "xml",
                    "tool": "health_claim_source_preparation",
                    "email": email,
                },
                timeout=(10, 60),
            )

            response.raise_for_status()
            raw = response.content

            filename = f"pubmed-cache/{key}.xml"
            path = folder / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)

            root = ET.fromstring(raw)

            if (
                root.tag != "PubmedArticleSet"
                or root.find(".//ERROR") is not None
            ):
                raise ValueError(
                    "Unexpected PubMed XML or service error document"
                )

            records = parse_pubmed(raw)
            found = [
                record["pmid"]
                for record in records
            ]

            if (
                len(found) != len(set(found))
                or not set(found).issubset(ids)
            ):
                raise ValueError(
                    "PubMed returned duplicate or unexpected PMIDs"
                )

            state["cache"][key] = {
                "file": filename,
                "sha256": sha(raw),
                "retrieved_at": {
                    record["pmid"]: record["retrieved_at"]
                    for record in records
                },
            }

            event["status"] = "fetched"
            return records

        except (
            requests.RequestException,
            ET.ParseError,
            ValueError,
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

            if (
                attempt == 1
                or code in (400, 401, 403, 404, 429)
            ):
                raise

            time.sleep(1)

        finally:
            dump(folder / "preparation-state.json", state)

    raise RuntimeError(
        "PubMed attempt budget exhausted; "
        "inspect preparation-state.json"
    )


def run(args):
    worklist = Path(args.worklist).read_bytes()
    evidence = Path(args.evidence).read_bytes()

    jobs = load_jobs(worklist, evidence)

    ids = sorted({
        pmid
        for job in jobs
        for pmid in job["pmids"]
    })

    if ids and not args.email:
        raise ValueError(
            "Set NCBI_EMAIL to your contact email "
            "before fetching PubMed"
        )

    folder = Path(args.out_dir).resolve()

    binding = {
        "profile": "pending-claim-sources-v1",
        "worklist_sha256": sha(worklist),
        "evidence_sha256": sha(evidence),
        "script_sha256": sha(Path(__file__).read_bytes()),
    }

    with exclusive_run(folder):
        state_path = folder / "preparation-state.json"

        state = (
            json.loads(state_path.read_bytes())
            if state_path.exists()
            else {
                "binding": binding,
                "events": [],
                "cache": {},
            }
        )

        if state["binding"] != binding:
            raise ValueError(
                "Inputs/code changed; "
                "choose a new output directory"
            )

        for event in state["events"]:
            if event["status"] == "started":
                event["status"] = "interrupted_unknown"

        dump(state_path, state)

        records = {}

        for start in range(0, len(ids), 50):
            batch = ids[start:start + 50]

            for record in fetch_batch(
                folder,
                state,
                batch,
                args.email,
            ):
                records[record["pmid"]] = record

            if start + 50 < len(ids):
                time.sleep(0.4)

        summary = []

        for job in jobs:
            claim = job["claim"]
            cid = claim["claim_id"]
            directory = folder / cid

            sources = {}
            active = []
            missing = []
            empty = []
            hashes = {}

            for pmid in job["pmids"]:
                if pmid not in records:
                    missing.append(pmid)
                    continue

                record = records[pmid]
                text = record["text"]

                sid = "S" + sha(
                    (record["url"] + "\n" + text).encode()
                )[:16]

                relative = f"sources/{sid}.txt"
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(text.encode())

                hashes[sid] = sha(text.encode())

                sources[sid] = {
                    key: value
                    for key, value in record.items()
                    if key != "text"
                }

                sources[sid].update(
                    id=sid,
                    text_file=relative,
                    text_sha256=hashes[sid],
                    origin="existing_retrieval_candidate_refreshed",
                    characters=len(text),
                )

                if len(clean(text)) >= 12:
                    active.append(sid)
                else:
                    empty.append(pmid)

            status = (
                "source_snapshots_prepared"
                if active
                else "retrieval_or_source_access_required"
            )

            claim_binding = {
                **binding,
                "source_hashes": hashes,
            }

            limitations = [
                "Existing candidate pool only; no new literature "
                "search or relevance assessment.",
                "PubMed abstracts are not full texts. "
                "Missing abstracts remain unavailable.",
                "No official guidance was fetched or assessed "
                "in this preparation step.",
            ]

            dump(
                directory / "state.json",
                {
                    "claim": {
                        **claim,
                        "input_limitations": limitations,
                    },
                    "binding": claim_binding,
                    "sources": sources,
                    "active_sources": active,
                    "phase": "sources_prepared",
                    "counts": {"model_calls": 0},
                    "model_events": [],
                    "unresolved_pmids": missing,
                    "empty_abstract_pmids": empty,
                },
            )

            entry = {
                "claim_id": cid,
                "status": status,
                "candidate_pmids": len(job["pmids"]),
                "usable_abstracts": len(active),
                "unresolved_pmids": missing,
                "empty_abstract_pmids": empty,
                "structured_outcome_missing": not bool(
                    claim.get("outcome")
                ),
                "input_issue": job["input_issue"],
                "source_run": str(directory),
                "scientific_verification_status": "not_started",
                "verdict": None,
            }

            summary.append(entry)

            print(
                cid,
                "|",
                status,
                "| usable abstracts:",
                len(active),
            )

        dump(
            folder / "batch-preparation.json",
            {
                "binding": binding,
                "claims": summary,
                "unique_requested_pmids": len(ids),
                "model_calls": 0,
                "scope": (
                    "Source preparation only; no suitability "
                    "decisions, scientific verification or adjudication."
                ),
            },
        )

        print("MODEL CALLS: 0")
        print("SUMMARY:", folder / "batch-preparation.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--worklist",
        default="data/processed/project-claim-status.json",
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument(
        "--out-dir",
        default="data/processed/verification-batch-v1",
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("NCBI_EMAIL"),
    )

    run(parser.parse_args())