from google import genai #type:ignore

client = genai.Client()

response = client.interactions.create(
    model="gemini-3.8-flash",
    input="Say hello in one line."
)

print(response.output_text)