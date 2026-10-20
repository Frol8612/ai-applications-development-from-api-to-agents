import base64
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/reference/resources/images/methods/edit
# ---
# Request (multipart/form-data, NOT json):
# curl -X POST "https://api.openai.com/v1/images/edits" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -F "model=gpt-image-1" \
#     -F "image=@logo.png" \
#     -F "prompt=Add magical sparkles and glowing aura around the logo"
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": "Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO..."
#     }
#   ]
# }

#TODO:
# You need to edit an existing image with `gpt-image-2` model:
#   - Take a local image (e.g. 'logo.png') and a prompt describing the edit
#   - Send it to the OpenAI images edit API
#   - Decode the returned base64 image and save it locally
# ---
# Hints:
#   - Use /v1/images/edits endpoint
#   - The request must be 'multipart/form-data' (NOT json) — pass the image as a file and the prompt/model as form fields
#   - The edited image will be returned in base64 format


def main(model_name: str, image_path: str, prompt: str, **kwargs):
    url = f"{OPENAI_HOST}/openai/deployments/{model_name}/chat/completions"
    headers = {"api-key": OPENAI_API_KEY}

    image_path = Path(image_path)
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "custom_content": {
                    "attachments": [
                        {
                            "type": "image/png",
                            "title": image_path.name,
                            "data": image_base64,
                        }
                    ]
                },
            }
        ],
        "custom_fields": {
            "configuration": {
                **kwargs,
                "output_format": "png",
            }
        },
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()

    message = response.json()["choices"][0]["message"]
    attachments = message.get("custom_content", {}).get("attachments", [])
    image = next(
        (item for item in attachments if item.get("type") == "image/png"),
        None,
    )

    if image is None:
        raise RuntimeError(
            f"The model did not return a PNG attachment. Response: {message}"
        )

    if image.get("data"):
        image_bytes = base64.b64decode(image["data"], validate=True)
    elif image.get("url"):
        file_url = image["url"]

        if file_url.startswith("files/"):
            file_url = f"{OPENAI_HOST}/v1/{file_url}"
        else:
            file_url = urljoin(f"{OPENAI_HOST}/", file_url)

        parsed_url = urlparse(file_url)
        if parsed_url.scheme != "https":
            raise RuntimeError(f"Unsafe image URL: {file_url}")

        download_headers = (
            headers
            if parsed_url.netloc == urlparse(OPENAI_HOST).netloc
            else {}
        )

        download = requests.get(
            file_url,
            headers=download_headers,
            timeout=120,
            allow_redirects=False,
        )
        download.raise_for_status()
        if download.status_code != 200:
            raise RuntimeError(
                f"Unexpected download status: {download.status_code}"
            )

        image_bytes = download.content
    else:
        raise RuntimeError("The attachment contains neither data nor url.")

    filename = Path(f"edited_{datetime.now():%Y%m%d_%H%M%S_%f}.png")
    filename.write_bytes(image_bytes)
    print(f"Edited image saved as {filename}")


main(
    model_name="gpt-image-2-2026-04-21",
    image_path="logo.png",
    prompt=(
        "Transform this DIALX Community logo by adding magical sparkles, "
        "glowing stars, and a soft mystical aura around the letters. "
        "Keep the original text and shape clearly readable."
    ),
)
