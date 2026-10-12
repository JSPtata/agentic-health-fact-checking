import json
import sys
import os
from google import genai #type:ignore

client=genai.Client()

input_file=sys.argv[1]

with open(input_file) as file:
    data=json.load(file)

video_id=os.path.splitext(os.path.basename(input_file))[0]

output_dir="data/processed/suitability"
os.makedirs(output_dir,exist_ok=True)

output_file=os.path.join(output_dir,video_id+".json")


all_assessments=[]

for item in data["claims"]:

    articles=[]

    for article in item["evidence"]:
        articles.append({
                "pmid":article.get("pmid"),
                "title":article.get("title"),
                "abstract":article.get("abstract")
            })
    

    claim_payload={
        "claim_id":item["claim_id"],
        "claim_text":item["claim_text"],
        "exposure":item.get("exposure"),
        "outcome":item.get("outcome"),
        "context":item.get("context"),
        "articles":articles
    }

    prompt = f"""
    You are a biomedical Evidence Suitability Assessment Agent.

    Your task is to evaluate whether each retrieved scientific article is
    SUITABLE for verifying the given health claim.

    You are NOT determining whether the health claim is true or false.
    You are determining whether each article provides relevant evidence
    that can meaningfully contribute to its verification.

    INPUT:
    {json.dumps(claim_payload, indent=2)}

    ASSESSMENT INSTRUCTIONS:

    For each article, independently evaluate the following:

    1. EXPOSURE MATCH
    - Does the article investigate the same behavior, intervention,
        substance, treatment, or exposure mentioned in the claim?
    - Related medical terminology or established synonyms may count.
    - Merely mentioning a related health topic is insufficient.

    2. OUTCOME MATCH
    - Does the article investigate the health outcome explicitly
        described in the claim?
    - Consider medically equivalent outcomes where justified.
    - Do not confuse related symptoms with identical outcomes.
    - If the claim has no explicit outcome, return null.

    3. CONTEXT MATCH
    - Does the article address the important conditions in the claim,
        including timing, dosage, population, frequency, or comparisons?
    - If the claim includes a comparison, check whether the article
        investigates the relevant alternatives.
    - If the claim has no specified context, return null.
    - If the abstract does not provide enough information to establish
        a match, return false and explain the uncertainty.

    SUITABILITY CLASSIFICATION:

    DIRECT:
    - The article investigates the central exposure and the relevant
    health outcome or comparison.
    - Important claim-specific conditions are sufficiently addressed.
    - The article can meaningfully support OR contradict the claim.

    PARTIAL:
    - The article investigates a meaningful part of the claim.
    - However, an important outcome, condition, comparison, or
    population is missing or differs from the claim.
    - The article may provide useful background or indirect evidence.

    IRRELEVANT:
    - The article does not meaningfully investigate the claim.
    - It may share terminology but study an unrelated phenomenon.
    - A high semantic similarity score does not establish relevance.

    SPECIAL RULES:

    1. Use ONLY the supplied article titles and abstracts.
    2. Do not use prior medical knowledge to invent article findings.
    3. Do not assume that an article supports a claim simply because
    it investigates the same topic.
    4. Distinguish direct evidence from background information.
    5. Preserve every PMID exactly as provided.
    6. Evaluate every article independently.
    7. Never invent experimental results, conclusions, or study details.
    8. Base the explanation on information explicitly present
    in the supplied title or abstract.
    9. If evidence is unclear or incomplete, prefer PARTIAL or IRRELEVANT
    rather than overstating suitability.
    10. Do not treat the absence of relevant evidence as proof that
        the original health claim is false.
    11. Do not use numerical semantic similarity as proof of suitability.
    12. Do not mistake a study protocol for completed experimental evidence.

    After evaluating all articles, assess whether the COMBINED evidence
    adequately addresses the claim.

    Set needs_more_evidence to true when important parts of the claim
    remain unaddressed or when no article is directly relevant.

    Specify exactly what additional scientific evidence is missing.

    Return ONLY valid JSON in this exact structure:

    {{
    "claim_id": "...",
    "assessments": [
        {{
        "pmid": "...",
        "exposure_match": true,
        "outcome_match": false,
        "context_match": false,
        "suitability": "direct",
        "reason": "Evidence-grounded explanation."
        }}
    ],
    "needs_more_evidence": true,
    "missing_evidence": "Specific scientific evidence still required."
    }}

    OUTPUT REQUIREMENTS:
    - exposure_match, outcome_match and context_match must be
    true, false, or null.
    - suitability must be exactly one of:
    "direct", "partial", "irrelevant".
    - Return one assessment for every input article.
    - Do not add or omit PMIDs.
    - If no additional evidence is needed, set missing_evidence to null.
    - Do not include Markdown formatting or code fences.
    """

    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    response_text=response.output_text.strip()
    response_text=response_text.replace("```json", "").replace("```", "")

    assessment=json.loads(response_text)

    if assessment["claim_id"] !=item["claim_id"]:
        raise ValueError("Claim ID mismatch")
    
    expected_pmids=set()
    
    for article in articles:
        expected_pmids.add(str(article["pmid"]))

    returned_pmids=[]

    for a in assessment["assessments"]:
        returned_pmids.append(str(a["pmid"]))

    if set(returned_pmids) != expected_pmids:
        raise ValueError("PMID mismatch: Missing or unexpected articles")

    if len(returned_pmids) != len(set(returned_pmids)):
        raise ValueError("Duplicate PMIDs found")
    
    allowed_labels={"direct","partial","irrelevant"}

    for a in assessment["assessments"]:
        if a["suitability"] not in allowed_labels:
            raise ValueError("Invalid suitability label")
        
        for field in ["exposure_match","outcome_match","context_match"]:
            if type(a[field]) not in (bool,type(None)):
                raise ValueError(f"Invalid value for {field}")
            
    if type(assessment["needs_more_evidence"]) is not bool:
        raise ValueError("needs_more_evidence must be a Boolean")
    
    missing=assessment["missing_evidence"]

    if missing is not None and not isinstance(missing,str):
        raise ValueError("missing_evidence must be a string or null")
    
    all_assessments.append(assessment)

    with open(output_file,"w") as f:
        json.dump({"claims":all_assessments},f,indent=4)

    print(json.dumps(assessment, indent=4))
    print("Validated assessments:", len(all_assessments))




print("Saved suitability results to:",output_file)