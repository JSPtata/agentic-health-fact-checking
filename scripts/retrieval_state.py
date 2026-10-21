import hashlib
import json
import os
import tempfile
from pathlib import Path

def claim_fingerprint(claim):
    specification={
        key:claim.get(key)
        for key in(
            "claim_id",
            "claim_text",
            "exposure",
            "outcome",
            "context"
        )
    }
    
    encoded=json.dumps(specification,sort_keys=True,ensure_ascii=False).encode("utf-8")
    
    return hashlib.sha256(encoded).hexdigest()

def state_path(claim_id,state_dir):
    filename=hashlib.sha256(claim_id.encode("utf-8")).hexdigest()
    return Path(state_dir)/f"{filename}.json"

def load_state(claim,state_dir="data/processed/retrieval_state"):
    path=state_path(claim["claim_id"],state_dir)
    fingerprint=claim_fingerprint(claim)
    
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            state=json.load(handle)
            
        if state.get("schema_version") !=1:
            raise ValueError("Unsupported retrieval-state version.")
        
        if state.get("claim_fingerprint") != fingerprint:
            raise ValueError(
                "The claim has changed.Use a seperate state directory"
                "for the new experiment."
            )
        return state
    
    
    return{
        "schema_version":1,
        "claim_id":claim["claim_id"],
        "claim_fingerprint":fingerprint,
        "status":"not_started",
        "rounds":[],
        "seen_pmids":[],
        "unresolved_gaps":[],
        "stop_reason":None,
    }
    
def save_state(state,state_dir="data/processed/retrieval_state"):
    path=state_path(state["claim_id"],state_dir)
    path.parent.mkdir(parents=True,exist_ok=True)
    
    temporary_path=None
    
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        )as handle:
            temporary_path=Path(handle.name)
            
            json.dump(state,handle,indent=2,ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
            
        os.replace(temporary_path,path)
        
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)