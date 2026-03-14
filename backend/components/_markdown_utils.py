from dataclasses import dataclass
from functools import lru_cache

from bs4 import BeautifulSoup
from markdown import markdown as render_markdown

_HEADINGS = tuple(f"h{level}" for level in range(1, 7))
_MARKDOWN_EXTENSIONS = ("extra", "tables", "fenced_code", "sane_lists")


@dataclass(frozen=True, slots=True)
class MarkdownInspection:
    headings: tuple[tuple[int, str], ...]
    chunk_type: str


@lru_cache(maxsize=2048)
def inspect_markdown(content: str) -> MarkdownInspection:
    soup = _parse_markdown(content)

    headings = tuple(
        (int(tag.name[1]), tag.get_text(" ", strip=True))
        for tag in soup.find_all(_HEADINGS)
        if tag.get_text(" ", strip=True)
    )

    if soup.find(["pre", "code"]):
        chunk_type = "code"
    elif soup.find("table"):
        chunk_type = "table"
    elif soup.find("img"):
        chunk_type = "figure_caption"
    elif soup.find(["ul", "ol"]):
        chunk_type = "list"
    else:
        chunk_type = "text"

    return MarkdownInspection(headings=headings, chunk_type=chunk_type)


def extract_title(content: str) -> str:
    inspection = inspect_markdown(content or "")
    for level, title in inspection.headings:
        if level == 1:
            return title
    return ""


@lru_cache(maxsize=2048)
def _parse_markdown(content: str) -> BeautifulSoup:
    html = render_markdown(
        content or "",
        extensions=list(_MARKDOWN_EXTENSIONS),
        output_format="html5",
    )
    return BeautifulSoup(html, "html.parser")
