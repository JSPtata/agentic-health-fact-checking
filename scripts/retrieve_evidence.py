import json
import sys
import os
import requests #type:ignore 
from Bio import Entrez #type:ignore
from Bio.Entrez import efetch, read #type:ignore
from itertools import combinations

from google import genai #type:ignore
from google.genai import types #type:ignore

from sentence_transformers import SentenceTransformer #type:ignore
from sklearn.metrics.pairwise import cosine_similarity #type:ignore

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=3,
            initial_delay=2,
            max_delay=10,
            exp_base=2,
            jitter=1
        )
    )
)

model=SentenceTransformer(
    "pritamdeka/S-PubMedBert-MS-MARCO"
)

Entrez.email = "tatajspranathi@gmail.com" 

input_file=sys.argv[1]

video_id=input_file.split("/")[-1].replace(".json","")

import spacy #type:ignore

nlp = spacy.load("en_core_web_sm")

with open(input_file,"r") as file:
    data=json.load(file)

claims=data["claims"]


def extract_concepts(claim):
    doc=nlp(claim)

    concepts=[]

    root=list(doc.sents)[0].root

    if root.pos_=="VERB":
        concepts.append(root.lemma_)

    for chunk in doc.noun_chunks:
        concept = chunk.root.lemma_

        if concept:
            concepts.append(concept)

    return concepts



def build_query_variants(concepts):
    size=len(concepts)
    queries=[]
    
    while size>=2:
        for j in combinations(concepts,size):
            query=" AND ".join(j)
            queries.append(query)
        size-=1

    return queries


def generate_queries_with_gemini(claim):
    prompt = f"""
    You are a biomedical information retrieval specialist.

    Generate exactly 3 PubMed search queries for this health claim:

    {claim}

    Rules:
    - Preserve the main medical concepts.
    - Include important outcomes and comparisons.
    - Use biomedical synonyms where appropriate.
    - Use PubMed-compatible AND/OR operators.
    - Do not invent additional medical conditions.
    - Return only a JSON array of 3 strings.

    Example:
    [
      "(screen time OR electronic devices) AND sleep quality",
      "(blue light OR smartphone use) AND bedtime",
      "screen exposure AND sleep duration"
    ]
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    queries = json.loads(response.text)

    if not isinstance(queries, list):
        raise ValueError("Expected a list of queries")

    if len(queries) != 3 or not all(
        isinstance(query, str) and query.strip()
        for query in queries
    ):
        raise ValueError("Invalid query output")

    return queries



def search_pubmed(query):
    url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params={
        "db":"pubmed",
        "term":query,
        "retmax":5,
        "retmode":"json"
    }

    response=requests.get(url,params=params)

    data = response.json()

    return data["esearchresult"]["idlist"]



def fetch_article_details(pmid):
    handle=efetch(
        db='pubmed',
        id=pmid,
        retmode='xml'
        )
    
    record=read(handle)
    handle.close()

    try:
        article=record["PubmedArticle"][0]["MedlineCitation"]["Article"]
        abstract=article['Abstract']['AbstractText'][0]
        return {
            "title":article["ArticleTitle"],
            "abstract":abstract
        }
    except (KeyError, IndexError):
        return None
    

def retrieve_candidates(query_variants):
    all_pmids = set()
    used_queries = []

    for query in query_variants[:3]:
        pmids=search_pubmed(query)

        if pmids:
            used_queries.append(query)

            for pmid in pmids:
                all_pmids.add(pmid)

    return {
        "queries": used_queries,
        "pmids": list(all_pmids)
    }

def fetch_all_articles(pmids):
    articles=[]

    for pmid in pmids:
        details=fetch_article_details(pmid)

        if details:
            details["pmid"]=pmid
            articles.append(details)

    return articles
    
def rerank_articles(claim, articles):
    if not articles:
        return []

    article_texts=[]
    
    for article in articles:
        text=article["title"]+" "+str(article["abstract"])
        article_texts.append(text)

    claim_embedding=model.encode([claim])
    article_embeddings=model.encode(article_texts,batch_size=8,show_progress_bar=False)
    scores=cosine_similarity(claim_embedding,article_embeddings).flatten()

    for i in range(len(articles)):
        articles[i]["relevance_score"]=float(scores[i])

    articles.sort(
        key=lambda x:x["relevance_score"],
        reverse=True
    )

    return articles

all_results=[]

for item in claims:
    claim = item["claim_text"]

    try:
        query_variants = generate_queries_with_gemini(claim)
        query_source = "gemini"

    except Exception as e:
        print("Gemini query generation failed:", e)

        concepts = extract_concepts(claim)
        query_variants = build_query_variants(concepts)
        query_source = "spacy_fallback"

    result = retrieve_candidates(query_variants)


    if result["pmids"]:
        articles = fetch_all_articles(result["pmids"])

        if articles:
            reranked_articles = rerank_articles(claim, articles)
            top_articles=reranked_articles[:3]
        else:
            top_articles=[]


        claim_result={
            "claim_text":claim,
            "queries_used":result["queries"],
            "query_source": query_source,
            "evidence":top_articles,

            "claim_id": item["claim_id"],
            "exposure": item.get("exposure"),
            "outcome": item.get("outcome"),
            "context": item.get("context")
        }
    else:
        claim_result={
            "claim_text":claim,
            "queries_used":[],
            "query_source": query_source,
            "evidence":[],

            "claim_id": item["claim_id"],
            "exposure": item.get("exposure"),
            "outcome": item.get("outcome"),
            "context": item.get("context")
        }
    
    all_results.append(claim_result)


output_data={
    "video_id":video_id,
    "claims":all_results
}

os.makedirs("data/processed/evidence",exist_ok=True)

output_file=os.path.join(
    "data/processed/evidence",video_id+".json")

with open(output_file,"w") as file:
    json.dump(output_data,file,indent=4)

print("Saved evidence to:",output_file)
