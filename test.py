import requests
import re

# 1. Fetch the page
URL = "https://wordletools.azurewebsites.net/weightedbottles"
resp = requests.get(URL)
resp.raise_for_status()

# 2. Extract all 5‑letter uppercase words
words = re.findall(r'\b([A-Z]{5})\b', resp.text)

# 3. Convert to lowercase, dedupe & sort
unique_words = sorted(set(w.lower() for w in words))

# 4. Write them out
with open("wordle_words.txt", "w") as f:
    f.write("\n".join(unique_words))

print(f"Scraped {len(unique_words)} words → wordle_words.txt")