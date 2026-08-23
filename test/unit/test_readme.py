import re
from pathlib import Path

import pytest


def extract_python_blocks(markdown_text: str) -> list[str]:
    """Extract Python code blocks from markdown text."""
    pattern = r"```python\s*\n(.*?)```"
    return re.findall(pattern, markdown_text, re.DOTALL)


def test_readme_code_examples_execute() -> None:
    """Execute all Python code snippets found in README.md."""
    readme_path = Path(__file__).resolve().parents[2] / "README.md"
    assert readme_path.exists(), f"README.md not found at {readme_path}"

    content = readme_path.read_text(encoding="utf-8")
    blocks = extract_python_blocks(content)
    assert len(blocks) > 0, "No python code blocks found in README.md"

    for i, code_block in enumerate(blocks):
        namespace: dict[str, object] = {}
        try:
            exec(code_block, namespace)  # noqa: S102 -- executing readme snippets in test harness
        except Exception as e:
            pytest.fail(f"README.md Python code block #{i + 1} failed to execute:\n{code_block}\nError: {e}")


def test_readme_relative_links_exist() -> None:
    """Verify relative file links in README.md resolve to existing files on disk."""
    repo_root = Path(__file__).resolve().parents[2]
    readme_path = repo_root / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
    for text, target in links:
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        file_part = target.split("#")[0]
        if not file_part:
            continue
        resolved = (repo_root / file_part).resolve()
        assert resolved.exists(), (
            f"Broken link in README.md for '{text}': target '{target}' does not exist at {resolved}"
        )
