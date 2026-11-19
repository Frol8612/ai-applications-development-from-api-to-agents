import json

import requests


class EmbeddingsClient:
    _endpoint: str
    _api_key: str
    _model_name: str

    def __init__(self, endpoint: str, model_name: str, api_key: str):
        if not endpoint or endpoint.strip() == "":
            raise ValueError("Endpoint cannot be null or empty")
        if not model_name or model_name.strip() == "":
            raise ValueError("Model name cannot be null or empty")
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = endpoint
        self._api_key = "Bearer " + api_key
        self._model_name = model_name

    def get_embeddings(
            self, inputs: str | list[str],
            dimensions: int,
            print_response: bool = False
    ) -> dict[int, list[float]]:
        """
        Generate dict of indexed embeddings:
            inputs[0](text) -> [0][embedding]
            inputs[1](text) -> [1][embedding]
            ...

        Args:
            inputs: input text, can be singular string or list of strings
            dimensions: number of dimensions
            print_response: to print response in chat or not
        """
        if not isinstance(inputs, (str, list)):
            raise TypeError("Inputs must be a string or a list of strings")
        if isinstance(inputs, list) and (
            not inputs or any(not isinstance(item, str) for item in inputs)
        ):
            raise ValueError("Inputs list must contain at least one string")
        if isinstance(inputs, str) and not inputs.strip():
            raise ValueError("Input text cannot be empty")
        if dimensions <= 0:
            raise ValueError("Dimensions must be greater than zero")

        request_data = {
            "input": inputs,
            "model": self._model_name,
            "dimensions": dimensions,
        }
        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }

        response = requests.post(
            url=self._endpoint,
            headers=headers,
            json=request_data,
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        response_data = response.json()
        if print_response:
            print(json.dumps(response_data, indent=2))

        data = response_data.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError("Embeddings response does not contain data")

        embeddings: dict[int, list[float]] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Embeddings response contains an invalid item")

            index = item.get("index")
            embedding = item.get("embedding")
            if not isinstance(index, int) or not isinstance(embedding, list):
                raise ValueError("Embeddings response item is missing index or embedding")
            if len(embedding) != dimensions:
                raise ValueError(
                    f"Embedding {index} has {len(embedding)} dimensions; "
                    f"expected {dimensions}"
                )
            embeddings[index] = embedding

        expected_count = 1 if isinstance(inputs, str) else len(inputs)
        if len(embeddings) != expected_count:
            raise ValueError(
                f"Embeddings response contains {len(embeddings)} items; "
                f"expected {expected_count}"
            )

        return embeddings
