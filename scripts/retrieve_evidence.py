import json
import sys
import os
import requests #type:ignore 
from Bio import Entrez #type:ignore
from Bio.Entrez import efetch, read #type:ignore
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Entrez.email = "tatajspranathi@gmail.com" 

input_file=sys.argv[1]

video_id=input_file.split("/")[-1].replace(".json","")

import spacy #type:ignore

nlp = spacy.load("en_core_web_sm")

with open(input_file,"r") as file:
    data=json.load(file)

claims=[]

for item in data["claims"]:
    claims.append(item["claim_text"])



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
    article_texts=[]
    
    for article in articles:
        article_text=article["title"]+" "+article["abstract"]
        article_texts.append(article_text)

    vectorizer=TfidfVectorizer()
    tfidf_matrix=vectorizer.fit_transform(article_texts)

    claim_vector=vectorizer.transform([claim])
    scores=cosine_similarity(claim_vector,tfidf_matrix).flatten()

    for i in range(len(articles)):
        articles[i]["relevance_score"]=float(scores[i])

    articles.sort(
        key=lambda x:x["relevance_score"],
        reverse=True
    )

    return articles

all_results=[]

for claim in claims:

    concepts=extract_concepts(claim)
    query_variants = build_query_variants(concepts)

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
            "evidence":top_articles
        }
    else:
        claim_result={
            "claim_text":claim,
            "queries_used":[],
            "evidence":[]
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
