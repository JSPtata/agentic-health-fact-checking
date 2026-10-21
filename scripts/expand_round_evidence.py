import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

from retrieval_state import load_state, save_state
from retrieve_evidence import fetch_all_articles

def expand_round(run_directory):
    run_dir=Path(run_directory).resolve()
    
    initial = json.loads(
        (run_dir / "initial-evidence.json").read_text(encoding="utf-8")
    )
    
    if len(initial["claims"]) != 1:
        raise ValueError("Expected a single-claim experiment.")

    state_dir = run_dir / "data/processed/retrieval_state"
    state = load_state(initial["claims"][0], state_dir)
    
    if state["status"] != "needs_planning":
        raise ValueError(
            "Apply the current reassessment before expanding evidence."
        )
        
    current_round = state["rounds"][-1]

    input_path = Path(current_round["assessment_input_file"])
    if not input_path.is_absolute():
        input_path = run_dir / input_path

    input_bytes = input_path.read_bytes()
    input_hash = hashlib.sha256(input_bytes).hexdigest()

    if input_hash != current_round["assessment_input_sha256"]:
        raise ValueError("The saved evidence input has changed.")

    data = json.loads(input_bytes)
    claim = data["claims"][0]

    if claim["claim_id"] != state["claim_id"]:
        raise ValueError("Claim ID mismatch.")

    existing = {
        str(article["pmid"]): article
        for article in claim["evidence"]
    }

    candidates = set(existing)

    for record in state["rounds"]:
        candidates.update(
            str(pmid)
            for pmid in record.get("new_pmids", [])
        )

    if len(candidates) > 40:
        raise ValueError(
            "More than 40 candidates. Review the screening budget first."
        )

    pending = sorted(candidates - set(existing))

    if not pending:
        print("No omitted candidates to recover. State unchanged.")
        return

    print("Previously retained articles:", len(existing), flush=True)
    print("Omitted candidates to fetch:", len(pending), flush=True)

    recovered = fetch_all_articles(pending)

    recovered_by_id = {
        str(article["pmid"]): article
        for article in recovered
    }

    if not set(recovered_by_id).issubset(set(pending)):
        raise ValueError("Fetcher returned an unexpected PMID.")

    unresolved = sorted(set(pending) - set(recovered_by_id))

    if not recovered_by_id:
        print("No articles recovered. State unchanged.")
        print("Unresolved PMIDs:", unresolved)
        return

    combined = list({
        **existing,
        **recovered_by_id,
    }.values())

    claim["evidence"] = combined

    data["retrieval_coverage"] = {
        "candidate_pmids": sorted(candidates),
        "unresolved_pmids": unresolved,
        "selection": "all_available_candidates_without_rank_cutoff",
    }

    output_dir = (
        run_dir
        / "data/processed/candidate_screening"
        / uuid4().hex
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    output_file = output_dir / "assessment-input.json"

    output_bytes = json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    output_file.write_bytes(output_bytes)


    current_round.setdefault("assessment_history", []).append({
        "input_file": current_round["assessment_input_file"],
        "input_sha256": current_round["assessment_input_sha256"],
        "assessment_file": current_round.get("assessment_file"),
        "assessment_sha256": current_round.get("assessment_sha256"),
        "suitability_counts": current_round.get("suitability_counts"),
    })

    current_round["assessment_input_file"] = str(output_file)
    current_round["assessment_input_sha256"] = (
        hashlib.sha256(output_bytes).hexdigest()
    )
    current_round["unresolved_pmids"] = unresolved
    current_round["status"] = "awaiting_reassessment"

    state["status"] = "awaiting_reassessment"
    save_state(state, state_dir)

    print("Articles now available for screening:", len(combined))
    print("Unresolved PMIDs:", unresolved)
    print("New assessment input:", output_file)
    print("Previous input and assessment files were preserved.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python3 scripts/expand_round_evidence.py "
            "<experiment-directory>"
        )

    expand_round(sys.argv[1])
    