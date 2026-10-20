import base64

from commons.constants import OPENAI_CHAT_COMPLETIONS_ENDPOINT
from t3_content_generation._openai_client import OpenAIClientT3


# https://developers.openai.com/api/docs/guides/images-vision?format=url&lang=curl
# https://developers.openai.com/api/docs/guides/images-vision?format=base64-encoded

#TODO:
# You need to analyse these 2 images:
#   - https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg
#   - in this folder we have 'logo.png', load it as encoded data (see documentation)
# ---
# Hints:
#   - Use OpenAIClientT3 to connect to OpenAI API
#   - Use /v1/chat/completions endpoint
#   - Function to encode image to base64 you can find in documentation
# ---
# In the end load both images (url and base64 encoded 'logo.png'), ask "Generate poem based on images" and se what will happen?


def _encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main(model_name: str, img_urls: list[str], request: str = "What's in this image/s?"):
    client = OpenAIClientT3(OPENAI_CHAT_COMPLETIONS_ENDPOINT)

    img_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": img_url,
            }
        }
        for img_url in img_urls
    ]

    client.call(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": request
                },
                *img_content,
            ],
        }],
    )


main(
    model_name='gpt-4.1-mini-2025-04-14',
    img_urls=[
        'https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg',
        f"data:image/jpeg;base64,{_encode_image('logo.png')}"
    ],
    request="Poem based on images",
)