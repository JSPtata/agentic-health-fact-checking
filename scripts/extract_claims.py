import json
import sys
import os
from google import genai #type:ignore

client=genai.Client()

input_file=sys.argv[1]

with open(input_file) as file:
    data=json.load(file)

    speech_text=data["text"]

    ocr_text=""
    for item in data["on_screen_text"]:
        ocr_text+=item["text"]+"\n"

    combined_text=speech_text+"\n"+ocr_text

prompt = f"""
    Extract factual health claims from the text below.

    Then decompose compound claims into atomic claims.

    Rules for atomic decomposition:
    - Each atomic claim must contain exactly one factual idea.
    - Split only when the original statement contains multiple distinct facts.
    - Do not create two claims that mean the same thing.
    - Do not split a claim only because it contains different adjectives such as
    "rapid", "significant", or "large".
    - Preserve important qualifiers from the original statement.
    - Do not invent claims that are not explicitly stated.
    - Each claim must be independently verifiable.

    - Preserve the exact meaning of the original statement.
    - Do not infer the opposite of a recommendation.
    - Keep comparative or contrastive statements together when splitting them would change or exaggerate their original meaning.
    - Do not introduce a cause, effect, or health outcome unless it is explicitly stated.
    - Preserve words such as "may", "should", "best", "avoid", and comparisons.

    - Ignore greetings, filler, opinions, advertisements, and obvious OCR noise.

    For each extracted atomic claim, also identify:

    - exposure: The behavior, substance, treatment, or intervention being discussed.
    - outcome: The health effect explicitly claimed.
    - context: Important timing, dosage, population, conditions, or comparisons.

    Rules:
    - Extract only information explicitly present in the claim.
    - Do not invent an outcome or assume a causal relationship.
    - If a component is not explicitly stated, return null.
    - Preserve important comparisons and qualifiers.
    - For comparative claims, retain both alternatives in the context.
    - Do not change the original claim_text.

    Return only this JSON format:

    {{
        "claims": [
            {{
                "claim_text": "...",
                "exposure": "...",
                "outcome": "...",
                "context": "..."
            }}
        ]
    }}

    Text:
    {combined_text}
    """

response=client.interactions.create(
    model="gemini-3.8-flash",
    input=prompt
)

video_id=os.path.splitext(os.path.basename(input_file))[0]

os.makedirs("data/processed/claims",exist_ok=True)

output_file=os.path.join("data/processed/claims",video_id+".json")

response_text = response.output_text.strip()

response_text = response_text.replace("```json", "")
response_text = response_text.replace("```", "")

claims_data = json.loads(response_text)

for i, claim in enumerate(claims_data["claims"], start=1):
    claim["claim_id"] = f"{video_id}-c{i:03d}"

with open(output_file,"w") as json_file:
    json.dump(claims_data,json_file,indent=4)

print("Saved to: "+output_file)
print(json.dumps(claims_data, indent=4))