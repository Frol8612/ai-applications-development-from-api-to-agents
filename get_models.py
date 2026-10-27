import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST

response = requests.get(
    f"{OPENAI_HOST}/openai/models",
    headers={"Api-Key": OPENAI_API_KEY},
    timeout=30,
)
response.raise_for_status()

data = response.json()
for model in data.get("data", []):
    print(model.get("id"))