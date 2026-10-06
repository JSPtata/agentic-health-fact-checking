import json
import sys
from google import genai #type:ignore

client=genai.Client()

input_file=sys.argv[1]

with open(input_file,"r") as file:
    data=json.load(file)

claims=[]

for item in data["claims"]:
    claims.append(item["claim_text"])

def make_query_search(claims):

    claims_text=""
    for i,claim in enumerate(claims):
        claims_text+=f"{i+1}.{claim}\n"
    
    prompt=f"""
    Convert each health claim below into a short PubMed search query.

    Keep only the important medical concepts.
    Use AND between different concepts.
    Use OR only for close synonyms if useful.
    Do not add explanations.

    Return ONLY this JSON format:

    {{
    "queries": [
        {{
        "claim": "...",
        "query": "..."
        }}
    ]
    }}

    Claims:
    {claims_text}
    """

    print("Sending claims to Gemini...")

    response=client.interactions.create( 
        model="gemini-3.8-flash",
        input=prompt
    )


    print("Gemini response received.")

    return response.output_text.strip()


queries=make_query_search(claims)

print(queries)