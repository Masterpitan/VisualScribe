import os
import json
import requests
from dotenv import load_dotenv

# Load key from .env file
load_dotenv()
api_key = os.getenv("META_API_KEY")

if not api_key:
    print("❌ Error: META_API_KEY not found in .env file!")
    exit()

# Example test call to an OpenAI-compatible/Meta inference endpoint
# (The exact URL and model name will be provided in your hackathon welcome docs)
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "muse-spark-1.3",
    "messages": [
        {"role": "user", "content": "Hello! Reply with 'API connection successful' if you can read this."}
    ],
    "max_tokens": 500
}

# Example endpoint URL from the hackathon dashboard
url = "https://api.meta.ai/v1/chat/completions"  

try:
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("✅ Success! Response from model:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"⚠️ API returned status code {response.status_code}:")
        print(response.text)
except Exception as e:
    print(f"❌ Connection error: {e}")