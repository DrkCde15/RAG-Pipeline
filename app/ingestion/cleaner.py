import unicodedata


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(char for char in text if char in "\n\t" or char.isprintable())
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()
