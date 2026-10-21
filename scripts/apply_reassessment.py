import hashlib
import json
import sys
from pathlib import Path

from retrieval_state import load_state,save_state

def apply_reassessment(input_file,result_file):
    input_bytes=Path(input_file).read_bytes()
    result_bytes=Path(result_file).read_bytes()
    
    evidence_input=json.loads(input_bytes)
    result=json.loads(result_bytes)
    
    claims=evidence_input["claims"]
    
    if len(claims)!=1:
        raise ValueError("Expected exactly one claim for this round.")
    
    claim=claims[0]
    state=load_state(claim)
    
    if state["status"]!="awaiting_reassessment":
        raise ValueError("This claim is not awaiting reassessment.")
    
    round_record=state["rounds"][-1]
  
    input_hash=hashlib.sha256(input_bytes).hexdigest()
    
    if round_record.get("assessment_input_sha256")!=input_hash:
        raise ValueError("Input does not match the saved retrieval round.")
    
    provenance = result.get("provenance", {})

    if provenance.get("status") != "complete":
        raise ValueError(
            "The suitability run is incomplete. "
            f"Received status: {provenance.get('status')!r}"
        )

    if provenance.get("input_sha256") != input_hash:
        raise ValueError(
            "Assessment belongs to a different evidence input."
        )

    results=result.get("claims",[])
    
    if len(results) != 1:
        raise ValueError("Expected exactly one claim assessment.")

    assessment=results[0]
    
    if assessment["claim_id"] != claim["claim_id"]:
        raise ValueError("Claim ID mismatch.")
    
    expected_pmids=[
        str(article["pmid"])
        for article in claim["evidence"]
    ]
    
    if not expected_pmids:
        raise ValueError("No evidence was supplied for reassessment.")
    
    if len(expected_pmids) != len(set(expected_pmids)):
        raise ValueError("Duplicate PMIDs in the evidence input.")
    
    articles=assessment["assessments"]
    returned_pmids=[str(article["pmid"]) for article in articles]
    
    if (
        len(returned_pmids) != len(expected_pmids)
        or set(returned_pmids) != set(expected_pmids)
    ):
        raise ValueError("Missing, duplicate, or unexpected assessed PMIDs.")
    
    allowed_labels={"direct","partial","irrelevant"}
    
    for article in articles:
        if article["suitability"] not in allowed_labels:
            raise ValueError("Invalid suitability label.")
        
        for field in ["exposure_match","outcome_match","context_match"]:
            if type(article[field]) not in (bool,type(None)):
                raise ValueError(f"Invalid value for {field}.")
            
    needs_more=assessment["needs_more_evidence"]
    missing=assessment["missing_evidence"]
    
    if type(needs_more) is not bool:
        raise ValueError("needs_more_evidence must be a Boolean.")
    
    if needs_more:
        if not isinstance(missing, str) or not missing.strip():
            raise ValueError("Further retrieval requires an explicit gap.")
    elif missing is not None:
        raise ValueError("missing_evidence must be null when no gap is reported.")
    
    round_record["status"]="reassessed"
    round_record["assessment_file"]=str(Path(result_file).resolve())
    round_record["assessment_sha256"]=hashlib.sha256(
        result_bytes
    ).hexdigest()
    
    round_record["suitability_counts"] = {
        label: sum(
            article["suitability"] == label
            for article in articles
        )
        for label in sorted(allowed_labels)
    }
    
    state["evidence"] = claim["evidence"]
    state["latest_assessment"] = assessment
    state["unresolved_gaps"] = [missing.strip()] if needs_more else [] #type:ignore
    state["status"] = (
        "needs_planning"
        if needs_more
        else "ready_for_sufficiency_review"
    )
    
    save_state(state)

    print("Reassessment accepted for:", claim["claim_id"])
    print("State:", state["status"])
    print("Suitability counts:", round_record["suitability_counts"])
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python3 scripts/apply_reassessment.py "
            "<assessment-input.json> <suitability-result.json>"
        )

    apply_reassessment(sys.argv[1], sys.argv[2])
