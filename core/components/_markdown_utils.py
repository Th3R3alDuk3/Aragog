from functools import lru_cache

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import BaseModel, ConfigDict

from core.models.vocabulary import ChunkType

_MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable("table")


class MarkdownInspection(BaseModel):
    model_config = ConfigDict(frozen=True)

    headings: tuple[tuple[int, str], ...]
    chunk_type: ChunkType


@lru_cache(maxsize=2048)
def inspect_markdown(content: str) -> MarkdownInspection:
    tokens = _MARKDOWN.parse(content or "")
    headings = tuple(_extract_headings(tokens))

    if _has_token_type(tokens, "fence", "code_block"):
        chunk_type = "code"
    elif _has_token_type(tokens, "table_open"):
        chunk_type = "table"
    elif _has_inline_child(tokens, "image"):
        chunk_type = "figure_caption"
    elif _has_token_type(tokens, "bullet_list_open", "ordered_list_open"):
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


def _extract_headings(tokens: list[Token]) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        if not token.tag.startswith("h") or len(token.tag) != 2 or not token.tag[1].isdigit():
            continue
        if index + 1 >= len(tokens):
            continue
        inline = tokens[index + 1]
        if inline.type != "inline":
            continue
        title = inline.content.strip()
        if not title:
            continue
        headings.append((int(token.tag[1]), title))
    return headings


def _has_token_type(tokens: list[Token], *token_types: str) -> bool:
    wanted = set(token_types)
    return any(token.type in wanted for token in tokens)


def _has_inline_child(tokens: list[Token], child_type: str) -> bool:
    for token in tokens:
        if token.type != "inline" or not token.children:
            continue
        if any(child.type == child_type for child in token.children):
            return True
    return False
