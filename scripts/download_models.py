import os

import httpx


def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    response = httpx.post(
        f"{base_url}/api/pull",
        json={"name": model, "stream": False},
        timeout=3600,
    )
    response.raise_for_status()
    print(f"Modelo Ollama disponível: {model}")


if __name__ == "__main__":
    main()
