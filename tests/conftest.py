from pathlib import Path

import pytest


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    path = tmp_path / "manual.txt"
    path.write_text("Conteúdo confiável para teste.\nMais uma linha.", encoding="utf-8")
    return path
