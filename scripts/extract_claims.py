import json
from google import genai #type:ignore

client=genai.Client()

with open("data/processed/transcripts/nutrition-01.json") as file:
    data=json.load(file)

    speech_text=data["text"]

    ocr_text=""
    for item in data["on_screen_text"]:
        ocr_text+=item["text"]+"\n"

    combined_text=speech_text+"\n"+ocr_text

    prompt = f"""
        Extract only factual health claims from the text below.

        Ignore:
        - greetings
        - filler
        - opinions
        - advertisements
        - non-health statements
        - obvious OCR noise

        Return only a numbered list of claims.

        Text:
        {combined_text}
        """
    
    response=client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    print(response.output_text)