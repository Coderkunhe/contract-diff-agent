import pdfplumber
from dataclasses import dataclass, field


@dataclass
class ContractPage:
    page_num: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)


@dataclass
class ContractDocument:
    file_path: str
    total_pages: int
    pages: list[ContractPage]
    full_text: str
    full_text_bilingual: str = ""


def extract_contract(file_path: str, keep_english: bool = False) -> ContractDocument:
    """Extract text and tables from a contract PDF.

    If keep_english is True, bilingual content is preserved as-is.
    If False (default), attempts to filter out English text for Chinese-only analysis,
    reducing token usage significantly for bilingual contracts.
    """
    pages: list[ContractPage] = []
    full_text_parts: list[str] = []
    bilingual_parts: list[str] = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []

            # Only keep tables with 2+ columns (actual data tables, not layout boxes)
            real_tables = [t for t in tables if t and len(t[0]) >= 2]

            pages.append(ContractPage(
                page_num=i + 1,
                text=text,
                tables=real_tables,
            ))

            if text:
                bilingual_parts.append(text)
                full_text_parts.append(_strip_english(text) if not keep_english else text)

    # Build full text (potentially Chinese-only)
    full_text = "\n\n".join(full_text_parts)
    full_text_bilingual = "\n\n".join(bilingual_parts)

    return ContractDocument(
        file_path=file_path,
        total_pages=len(pages),
        pages=pages,
        full_text=full_text,
        full_text_bilingual=full_text_bilingual,
    )


def _strip_english(text: str) -> str:
    """Remove English lines from bilingual contract text.

    Heuristic: a line is English if < 20% of its chars are CJK or punctuation.
    This preserves Chinese content while dropping the English translation side.
    """
    lines = text.split("\n")
    chinese_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            chinese_lines.append("")
            continue
        cjk_count = sum(1 for c in stripped if _is_cjk(c) or c in "，。、；：「」『』（）《》【】")
        ratio = cjk_count / max(len(stripped), 1)
        if ratio >= 0.15:
            chinese_lines.append(stripped)

    return "\n".join(chinese_lines)


def _is_cjk(c: str) -> bool:
    cp = ord(c)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK chars ~1 token each, others ~0.75 token."""
    cjk = sum(1 for c in text if _is_cjk(c))
    other = len(text) - cjk
    return int(cjk * 1.0 + other * 0.75)


def extract_table_text(tables: list[list[list[str]]]) -> str:
    """Flatten multi-column tables into markdown-like text for diff analysis."""
    parts: list[str] = []
    for table in tables:
        parts.append("--- TABLE ---")
        for row in table:
            cells = [str(c).strip() if c else "" for c in row]
            parts.append(" | ".join(cells))
        parts.append("--- END TABLE ---")
    return "\n".join(parts)
