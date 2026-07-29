from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrontmatterDocument:
    meta: dict[str, str]
    body: str


def parse_frontmatter(raw: str) -> FrontmatterDocument:
    """Parse a small YAML-frontmatter subset used by SKILL.md files."""
    text = raw or ""
    if not text.startswith("---\n"):
        return FrontmatterDocument(meta={}, body=text)

    end = text.find("\n---", 4)
    if end < 0:
        return FrontmatterDocument(meta={}, body=text)

    meta_block = text[4:end]
    body = text[end + len("\n---") :]
    if body.startswith("\n"):
        body = body[1:]

    meta: dict[str, str] = {}
    for line in meta_block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        if key:
            meta[key] = value
    return FrontmatterDocument(meta=meta, body=body)


def format_frontmatter(meta: dict[str, object], body: str) -> str:
    lines = ["---"]
    for key in sorted(meta):
        value = str(meta[key])
        if any(ch in value for ch in [":", "#", "\n", '"']):
            escaped = value.replace('"', '\\"')
            lines.append(f'{key}: "{escaped}"')
        else:
            lines.append(f"{key}: {value}")
    lines.extend(["---", "", (body or "").rstrip(), ""])
    return "\n".join(lines)
