from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    content: str
    chunk_index: int


def split_text(text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size deve ser positivo e 0 <= overlap < chunk_size")
    text = text.strip()
    if not text:
        return []

    chunks: list[TextChunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        content = text[start:end].strip()
        if content:
            chunks.append(TextChunk(content=content, chunk_index=len(chunks)))
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        start = next_start
    return chunks
