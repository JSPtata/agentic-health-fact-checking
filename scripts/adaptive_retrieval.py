import json
import sys
from google import genai #type:ignore
import requests #type:ignore
import os
from retrieve_evidence import fetch_all_articles, rerank_articles
from requests.adapters import HTTPAdapter #type:ignore
from urllib3.util.retry import Retry #type:ignore

client=genai.Client()

evidence_file=sys.argv[1]
suitability_file=sys.argv[2]

with open(evidence_file,"r") as f:
    evidence_data=json.load(f)

with open(suitability_file,"r") as f:
    suitability_data=json.load(f)

suitability_map={
    item["claim_id"]:item
    for item in suitability_data["claims"]
}

def refine_queries(claim):
    cache_dir = "data/processed/query_cache"
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(
        cache_dir, f"{claim['claim_id']}.json"
    )

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)

        if cached.get("missing_evidence") == claim["missing_evidence"]:
            print("Using cached queries for", claim["claim_id"], flush=True)
            return cached["queries"]
        
    prompt = f"""
    You are a biomedical literature search planning agent.

    Generate exactly 2 targeted PubMed search queries.

    Original claim: {claim["claim_text"]}
    Exposure: {claim.get("exposure")}
    Outcome: {claim.get("outcome")}
    Context: {claim.get("context")}
    Missing evidence: {claim["missing_evidence"]}

    Rules:
    - Address the specific evidence gap.
    - Use medically meaningful synonyms when appropriate.
    - Preserve important timing, comparisons and populations.
    - Use PubMed-compatible AND/OR syntax.
    - Do not invent study findings.
    - Avoid excessively restrictive queries.
    - Make the first query specific and the second broader.
    - Return only a JSON array of 2 query strings.
    """
    response=client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    response_text=response.output_text.strip()
    response_text=response_text.replace("```json", "").replace("```", "")

    queries=json.loads(response_text)

    if (
        not isinstance(queries, list)
        or len(queries) != 2
        or not all(isinstance(q, str) and q.strip() for q in queries)
    ):
        raise ValueError("Invalid refined queries")
    
    with open(cache_file, "w") as f:
        json.dump({
            "claim_id": claim["claim_id"],
            "missing_evidence": claim["missing_evidence"],
            "queries": queries
        }, f, indent=4)
    
    return queries

def search_pubmed(query,retmax=15):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    retries = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True
    )

    with requests.session() as session:
        session.mount(
            "https://",
            HTTPAdapter(max_retries=retries)
        )

        response = requests.get(
            url,
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": retmax,
                "sort": "relevance"
            },
            timeout=30
        )

        response.raise_for_status()
        result=response.json()["esearchresult"]

        return{
            "count":int(result["count"]),
            "pmids":result["idlist"]
        }

claims_needing_retrieval=[]

for claim in evidence_data["claims"]:
    claim_id=claim["claim_id"]

    assessment=suitability_map.get(claim_id)

    if assessment is None:
        print(f"No suitability assessment found for {claim_id}")
        continue

    needs_more_evidence = assessment["needs_more_evidence"]
    missing_evidence = assessment["missing_evidence"]

    print("\nCLAIM ID:",claim_id)
    print("CLAIM:", claim["claim_text"])
    print("NEEDS MORE EVIDENCE:", needs_more_evidence)
    print("MISSING EVIDENCE:", missing_evidence)

    if needs_more_evidence:
        claims_needing_retrieval.append({
            "claim_id": claim_id,
            "claim_text": claim["claim_text"],
            "exposure": claim.get("exposure"),
            "outcome": claim.get("outcome"),
            "context": claim.get("context"),
            "missing_evidence": missing_evidence,
            "existing_pmids": [
                str(article["pmid"])
                for article in claim["evidence"]
            ]
        })

if claims_needing_retrieval:
    test_claim=claims_needing_retrieval[0]

    print("\nGenerating refined queries...", flush=True)

    queries = refine_queries(test_claim)

    existing_pmids=set(test_claim["existing_pmids"])
    seen_pmids=existing_pmids.copy()

    all_new_pmids=[]

    for query in queries:

        print("\nSearching PubMed...", flush=True)

        result=search_pubmed(query)

        new_pmids=[
            pmid for pmid in result["pmids"]
            if pmid not in seen_pmids
        ]

        seen_pmids.update(new_pmids)
        all_new_pmids.extend(new_pmids)

        print("\nQUERY:", query)
        print("TOTAL MATCHES:", result["count"])
        print("NEW PMIDS:", new_pmids)

    
    print("\nFetching new articles....",flush=True)

    new_articles=fetch_all_articles(all_new_pmids)
    print("Articles fetched:",len(new_articles))

    ranked_articles=rerank_articles(
        test_claim["claim_text"],
        new_articles
    )

    top_articles=ranked_articles[:10]


    original_claim = next(
        item for item in evidence_data["claims"]
        if item["claim_id"] == test_claim["claim_id"]
    )

    combined_evidence = original_claim["evidence"] + top_articles

    combined_evidence = list({
        str(article["pmid"]): article
        for article in combined_evidence
    }.values())

    output_dir = "data/processed/adaptive_evidence"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"{test_claim['claim_id']}.json"
    )

    output_data = {
        "claim_id": test_claim["claim_id"],
        "claim_text": test_claim["claim_text"],
        "exposure": test_claim.get("exposure"),
        "outcome": test_claim.get("outcome"),
        "context": test_claim.get("context"),
        "missing_evidence": test_claim["missing_evidence"],
        "queries_used": queries,
        "retrieval_round": 1,
        "evidence": combined_evidence
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4, default=str)

    print("Adaptive evidence saved to:", output_file)

    assessment_input = {
        "video_id": evidence_data["video_id"],
        "claims": [
            {
                "claim_id": output_data["claim_id"],
                "claim_text": output_data["claim_text"],
                "exposure": output_data["exposure"],
                "outcome": output_data["outcome"],
                "context": output_data["context"],
                "evidence": output_data["evidence"]
            }
        ]
    }

    assessment_file = os.path.join(
        output_dir,
        f"{test_claim['claim_id']}-assessment-input.json"
    )

    with open(assessment_file, "w") as f:
        json.dump(assessment_input, f, indent=4, default=str)

    print("Suitability input saved to:", assessment_file)

    print("\nTOP 10 NEW ARTICLES:")

    for article in top_articles:
        print("\nPMID:",article["pmid"])
        print("TITLE:",article["title"])
        print("RELEVANCE SCORE:",article["relevance_score"])

print("\n----------------------------")
print("Total claims:", len(evidence_data["claims"]))
print("Claims requiring retrieval:", len(claims_needing_retrieval))