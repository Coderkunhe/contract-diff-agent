"""Clause tree extraction - Step ① of the contract diff pipeline.

Parses a contract's full text into a structured clause tree.
"""

import re
from dataclasses import dataclass, field

# Match Chinese numbered sections: 一、 二、 ... 十九、
_RE_L1 = re.compile(
    r"^([一二三四五六七八九十]{1,3})[、，,．. ]\s*(.+)"
)
# Match sub-clauses: （一） （二） or (一) (二)
_RE_L2 = re.compile(
    r"^[（(]([一二三四五六七八九十]{1,3})[）)]\s*(.+)"
)
# Match attachments: 附件一 附件二
_RE_ATTACHMENT = re.compile(
    r"^附件([一二三四五六七八九十]{1,2})\s*(.+)"
)
# Match table/index-like garbage lines (fees schedule tables)
_RE_TABLE_HEADER = re.compile(
    r"^(三|四|技术|服务|经营|大类|费率|年费|返还|二级|三级|备注|跨类目)"
)


@dataclass
class ClauseNode:
    id: str
    number: str  # e.g. "一", "（一）"
    title: str
    level: int  # 1 = chapter, 2 = sub-clause
    full_text: str
    page_start: int | None = None
    children: list["ClauseNode"] = field(default_factory=list)


@dataclass
class ContractTree:
    file_path: str
    total_pages: int
    clauses: list[ClauseNode]
    attachments: list[dict]  # {title, full_text}
    tables: list[dict]  # {page, caption, raw_text}
    full_text_sections: dict[str, str]  # clause_id -> raw text block


def build_clause_tree(
    full_text: str,
    pages_data: list[dict] | None = None,
) -> ContractTree:
    """Parse contract text into a structured clause tree.

    Args:
        full_text: The full contract text (Chinese-only, filtered).
        pages_data: Optional list of {page_num, text, tables} per page.

    Returns:
        ContractTree with nested clause structure.
    """
    lines = full_text.split("\n")
    clauses: list[ClauseNode] = []
    attachments: list[dict] = []
    current_l1: ClauseNode | None = None
    current_l2: ClauseNode | None = None
    buf: list[str] = []
    in_attachment = False
    attachment_buf: list[str] = []
    attachment_title = ""
    in_table_area = False
    l1_idx = 0

    def flush_to_clause():
        """Flush accumulated text buffer to current L2 (if exists) or current L1."""
        nonlocal buf
        if not buf:
            return
        text = "\n".join(buf).strip()
        buf = []
        if not text:
            return
        if current_l2:
            if current_l2.full_text:
                current_l2.full_text += "\n" + text
            else:
                current_l2.full_text = text
        elif current_l1:
            if current_l1.full_text:
                current_l1.full_text += "\n" + text
            else:
                current_l1.full_text = text

    for line in lines:
        s = line.strip()

        if not s:
            buf.append("")
            continue

        # Attachment detection
        att_m = _RE_ATTACHMENT.match(s)
        if att_m and not in_attachment:
            flush_to_clause()
            if current_l1:
                clauses.append(current_l1)
                current_l1 = None
            current_l2 = None
            in_attachment = True
            attachment_title = f"附件{att_m.group(1)} {att_m.group(2).strip()}"
            attachment_buf = [s]
            continue

        if in_attachment:
            if _RE_L1.match(s) or _RE_L2.match(s):
                attachments.append({
                    "title": attachment_title,
                    "full_text": "\n".join(attachment_buf).strip(),
                })
                attachment_buf = []
                attachment_title = ""
                in_attachment = False
            else:
                attachment_buf.append(s)
                continue

        # L1: must check BEFORE table detection to avoid false matches
        l1_m = _RE_L1.match(s)
        if l1_m:
            flush_to_clause()
            if current_l1:
                clauses.append(current_l1)

            l1_idx += 1
            current_l1 = ClauseNode(
                id=str(l1_idx),
                number=l1_m.group(1),
                title=l1_m.group(2).strip(),
                level=1,
                full_text="",
            )
            current_l2 = None
            buf = [s]
            continue

        # Skip table garbled lines (but only if not a clause header)
        if _RE_TABLE_HEADER.match(s) and len(s) < 10:
            in_table_area = True
            continue

        if in_table_area:
            if len(s) < 50 and any(w in s for w in ["技术", "费率", "年费", "扣点", "类目", "返还"]):
                continue
            else:
                in_table_area = False

        # L2
        l2_m = _RE_L2.match(s)
        if l2_m and current_l1 is not None:
            flush_to_clause()

            current_l2 = ClauseNode(
                id=f"{current_l1.id}.{len(current_l1.children) + 1}",
                number=f"（{l2_m.group(1)}）",
                title=l2_m.group(2).strip(),
                level=2,
                full_text="",
            )
            current_l1.children.append(current_l2)
            buf = [s]
            continue

        # Normal text
        buf.append(s)

    # Final flush
    flush_to_clause()

    if in_attachment and attachment_buf:
        attachments.append({
            "title": attachment_title,
            "full_text": "\n".join(attachment_buf).strip(),
        })

    if current_l1:
        clauses.append(current_l1)

    # Post-process: filter table artifacts (clauses from garbled table text)
    _TABLE_KEYWORDS = ["技术", "费率", "年费", "类目", "返还", "扣点", "积分"]
    clauses = [
        c for c in clauses
        if not (
            len(c.title) < 30
            and not c.children
            and any(kw in c.title for kw in _TABLE_KEYWORDS)
        )
    ]

    # Extract tables from page data
    tables: list[dict] = []
    if pages_data:
        for p in pages_data:
            for t in p.get("tables", []):
                if t and len(t[0]) >= 2:
                    tables.append({
                        "page": p["page_num"],
                        "raw_text": _flatten_table(t),
                    })

    # Build text sections lookup
    sections: dict[str, dict] = {}
    for clause in clauses:
        sections[clause.id] = {
            "number": clause.number,
            "title": clause.title,
            "level": clause.level,
            "full_text": clause.full_text,
        }
        for child in clause.children:
            sections[child.id] = {
                "number": child.number,
                "title": child.title,
                "level": child.level,
                "full_text": child.full_text,
            }

    return ContractTree(
        file_path="",
        total_pages=0,
        clauses=clauses,
        attachments=attachments,
        tables=tables,
        full_text_sections=sections,
    )


def _flatten_table(table: list[list[str | None]]) -> str:
    """Convert a multi-column table to readable text."""
    rows = []
    for row in table:
        cells = [str(c).strip() if c else "" for c in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def print_tree(tree: ContractTree):
    """Debug: print clause tree structure."""
    for c in tree.clauses:
        print(f"[{c.id}] {c.number}、{c.title} ({len(c.full_text)} chars)")
        for child in c.children:
            print(f"  [{child.id}] {child.number} {child.title} ({len(child.full_text)} chars)")
    if tree.attachments:
        print(f"\nAttachments: {len(tree.attachments)}")
        for a in tree.attachments:
            print(f"  {a['title']} ({len(a['full_text'])} chars)")
    if tree.tables:
        print(f"Tables: {len(tree.tables)}")
