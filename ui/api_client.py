from typing import Any

import httpx


class APIError(Exception):
    """Erro retornado pela API RAG."""


class RAGAPIClient:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(300.0, connect=10.0),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            try:
                detail = error.response.json().get("detail", error.response.text)
            except ValueError:
                detail = error.response.text
            raise APIError(f"API respondeu HTTP {error.response.status_code}: {detail}") from error
        return response.json()

    def health(self) -> dict[str, str]:
        return self._request("GET", "/health")

    def ingest_file(self, filename: str, content: bytes) -> dict[str, Any]:
        return self._request(
            "POST",
            "/documents/ingest",
            files={"file": (filename, content)},
        )

    def chat(self, question: str) -> dict[str, Any]:
        return self._request("POST", "/chat", json={"question": question})

    def list_documents(self) -> list[dict[str, Any]]:
        return self._request("GET", "/documents")["documents"]

    def delete_document(self, document_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/documents/{document_id}")
