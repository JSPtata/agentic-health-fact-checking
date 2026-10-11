import json
import sys
import os
import spacy #type:ignore
from sentence_transformers import SentenceTransformer  #type:ignore
from sentence_transformers import util #type:ignore

nlp=spacy.load("en_core_web_sm")
model = SentenceTransformer(
    "pritamdeka/S-PubMedBert-MS-MARCO"
)

def extract_keywords(text):
    doc=nlp(text)

    keywords={
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and token.pos_ in {"NOUN","PROPN","ADJ"}
    }
    return keywords

def extract_claim_components(claim):
    doc=nlp(claim)

    noun_phrases=[
        chunk.text.lower()
        for chunk in doc.noun_chunks
    ]

    outcomes=[]

    for token in doc:
        if token.dep_ in {"dobj","obj"}:
            outcomes.append(token.text)

    return {
        "noun_phrases":noun_phrases,
        "outcome_candidates":outcomes
    }


def semantic_component_match(component, article_text):
    if not component:
        return None

    component_embedding = model.encode(
        component,
        convert_to_tensor=True
    )

    article_embedding = model.encode(
        article_text,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        component_embedding,
        article_embedding
    )

    return round(float(similarity.item()), 3)


def assess_relevance(claim,article,components):
    claim_terms=extract_keywords(claim)

    title_terms=extract_keywords(str(article["title"]))
    abstract_terms=extract_keywords(str(article["abstract"]))

    if not claim_terms:
        if not claim_terms:
            return {
                "semantic_score": round(article.get("relevance_score", 0.0), 3),
                "concept_coverage": 0.0,
                "title_coverage": 0.0,
                "matched_terms": [],
                "missing_terms": [],
                "relevance_flag": "low",
                "suitability": {
                    "exposure_match": None,
                    "outcome_match": None,
                    "context_match": None,
                    "overall": "pending"
                }
            }
    
    all_terms=title_terms | abstract_terms

    matched=claim_terms & abstract_terms

    concept_coverage=len(matched)/len(claim_terms)

    title_coverage=len(claim_terms & title_terms)/len(claim_terms)

    article_text = str(article["title"]) + " " + str(article["abstract"])

    exposure_score = semantic_component_match(
        components.get("exposure"),
        article_text
    )

    outcome_score = semantic_component_match(
        components.get("outcome"),
        article_text
    )

    context_score = semantic_component_match(
        components.get("context"),
        article_text
    )

    if concept_coverage >=0.60 and title_coverage >= 0.25:
        flag="high"
    elif concept_coverage >=0.35:
        flag="review"
    else:
        flag="low"

    return {
        "semantic_score": round(article.get("relevance_score", 0.0), 3),

        "concept_coverage": round(concept_coverage, 3),
        "title_coverage": round(title_coverage, 3),

        "matched_terms": sorted(matched),
        "missing_terms": sorted(claim_terms - all_terms),

        "relevance_flag": flag,

        "suitability": {
            "exposure_match": exposure_score,
            "outcome_match": outcome_score,
            "context_match": context_score,
            "overall": "pending"
        }
    }

def main(input_file):
    with open(input_file,'r') as file:
        data=json.load(file)

        for item in data["claims"]:
            claim=item["claim_text"]

            components = {
                "exposure": item.get("exposure"),
                "outcome": item.get("outcome"),
                "context": item.get("context")
            }

            for article in item["evidence"]:
                article["relevance_review"]=assess_relevance(claim,article,components)

    os.makedirs("data/processed/assessed_evidence",exist_ok=True)

    output_file=os.path.join('data/processed/assessed_evidence',data["video_id"]+".json")

    with open(output_file,"w") as file:
        json.dump(data,file,indent=4)
    
    print("Saved assessed evidence to:",output_file)

    for item in data["claims"]:
        print("\nCLAIM:", item["claim_text"])
        print("COMPONENTS:", {
            "exposure": item.get("exposure"),
            "outcome": item.get("outcome"),
            "context": item.get("context")
        })

        for article in item["evidence"]:
            review = article["relevance_review"]

            print("SUITABILITY:", review["suitability"])

            print("TITLE:", article["title"])
            print("COVERAGE:", review["concept_coverage"])
            print("FLAG:", review["relevance_flag"])
            print("MISSING:", review["missing_terms"])


if __name__=="__main__":
    main(sys.argv[1])
