import requests
import json

url = "http://localhost:5001/predict"
payload = {
    "title": "Apple iPhone 12 Pro 128GB",
    "average_rating": 4.5,
    "num_reviews": 150,
    "current_price": 550,
    "screen_size": 6.1,
    "memory_gb": 128
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
except Exception as e:
    print(f"Connection failed: {e}")
