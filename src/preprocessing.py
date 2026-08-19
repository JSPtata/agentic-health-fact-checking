import re

def clean_text(text):
    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    sample = "Drinking lemon water cures ALL diseases!!!"

    cleaned = clean_text(sample)

    print("Original:", sample)
    print("Cleaned:", cleaned)
