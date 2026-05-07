"""Custom LangChain-compatible embedding client.

Provides :class:`ModalEmbeddings`, a drop-in ``Embeddings`` implementation
that calls any service exposing an ``/embeddings`` endpoint (OpenAI,
vLLM, Modal, LM Studio, etc.).

"""

from typing import List
import requests
from langchain_core.embeddings import Embeddings


class ModalEmbeddings(Embeddings):
    """LangChain ``Embeddings`` backed by an OpenAI-compatible HTTP API.

    The service must expose a ``POST /embeddings`` endpoint that accepts
    ``{"model": "…", "input": ["…"]}`` and returns the standard OpenAI
    response shape.

    Args:
        url: Base URL of the embedding service (e.g. ``"http://localhost:1234/v1"``).
        model_name: Model identifier(e.g. ``"intfloat/multilingual-e5-large-instruct"``).
        api_key: Optional bearer token for authenticated endpoints.
    """

    def __init__(self, url: str, model_name: str, api_key: str = None):
        self.url = url
        self.model_name = model_name
        self.api_key = api_key

    def embed(self, text: List[str]) -> List[List[float]]:
        """Embed a list of texts via the configured API.

        Newlines are replaced with spaces before sending, since most
        embedding models treat them as noise.

        Args:
            text: Strings to embed.

        Returns:
            list[list[float]]: One embedding vector per input string.

        Raises:
            requests.HTTPError: If the API returns a non-2xx status.
        """
        # Newlines degrade embedding quality for most models
        cleaned_text = [t.replace("\n", " ") for t in text]

        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.url}/embeddings",
            json={
                "model": self.model_name,
                "input": cleaned_text
            },
            headers=headers
        )

        response.raise_for_status()

        data = response.json()["data"]
        return [item["embedding"] for item in data]

    def embed_documents(self, text: List[str]) -> List[List[float]]:
        """Embed multiple documents (LangChain interface).

        Args:
            text: Document strings to embed.

        Returns:
            list[list[float]]: One embedding vector per document.
        """
        return self.embed(text)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (LangChain interface).

        Args:
            text: The query string.

        Returns:
            list[float]: The embedding vector for *text*.
        """
        return self.embed([text])[0]

    def get_model_name(self) -> str:
        """Return the model identifier used for embedding requests."""
        return self.model_name


if __name__ == "__main__":
    embeds = ModalEmbeddings(
        url="https://lfoppiano--intfloat-multilingual-e5-large-instruct-embed-5da184.modal.run/",
        model_name="intfloat/multilingual-e5-large-instruct"
    )

    print(embeds.embed(
        ["We are surrounded by stupid kids",
         "We are interested in the future of AI"]
    ))
