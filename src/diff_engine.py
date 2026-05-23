import difflib
import re
import json
from datetime import datetime, timezone

from .pdf_extractor import ContractDocument, estimate_tokens


def diff_contracts(
    v1: ContractDocument,
    v2: ContractDocument,
    agent_version: str = "0.1.0",
) -> dict:
    """Compare two contract documents and produce structured diff output.

    For v0.1, this uses line-level text diff (difflib).
    Future versions will use semantic comparison with LLM.
    """
    v1_lines = v1.full_text.splitlines(keepends=False)
    v2_lines = v2.full_text.splitlines(keepends=False)

    matcher = difflib.SequenceMatcher(None, v1_lines, v2_lines)
    opcodes = matcher.get_opcodes()

    changes: list[dict] = []
    added_count = 0
    removed_count = 0
    modified_count = 0
    diff_id = 0

    # Track blocks of changes for grouping
    current_block: dict | None = None

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            current_block = None
            continue

        diff_id += 1
        v1_snippet = "\n".join(v1_lines[i1:i2]).strip()
        v2_snippet = "\n".join(v2_lines[j1:j2]).strip()

        # Determine change type
        if tag == "insert":
            change_type = "added"
            added_count += 1
            v1_snippet = None
        elif tag == "delete":
            change_type = "removed"
            removed_count += 1
            v2_snippet = None
        else:
            change_type = "modified"
            modified_count += 1

        # Try to infer clause reference from surrounding context
        clause_ref = _infer_clause_ref(v1_lines, i1, i2) or _infer_clause_ref(v2_lines, j1, j2)

        changes.append({
            "id": f"diff-{diff_id:03d}",
            "change_type": change_type,
            "clause_ref_v1": clause_ref if change_type in ("modified", "removed") else None,
            "clause_ref_v2": clause_ref if change_type in ("modified", "added") else None,
            "clause_title": _summarize_change(v1_snippet or v2_snippet or ""),
            "brief": _brief_change(change_type, v1_snippet, v2_snippet),
            "v1_snippet": _truncate(v1_snippet, 500) if v1_snippet else None,
            "v2_snippet": _truncate(v2_snippet, 500) if v2_snippet else None,
            "risk_categories": [],
            "risk_level": "low",
            "risk_note": "",
            "attention_for": None,
            "is_favorable": None,
        })

    # Build meta
    meta = {
        "contract_v1": v1.file_path,
        "contract_v2": v2.file_path,
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "agent_version": agent_version,
        "token_estimate": {
            "v1_tokens": estimate_tokens(v1.full_text),
            "v2_tokens": estimate_tokens(v2.full_text),
            "total": estimate_tokens(v1.full_text) + estimate_tokens(v2.full_text),
        },
    }

    return {
        "meta": meta,
        "diff_summary": {
            "total_changes": diff_id,
            "added": added_count,
            "removed": removed_count,
            "modified": modified_count,
            "coverage_pct": _calculate_coverage(v1_lines, v2_lines, opcodes),
        },
        "changes": changes,
        "risk_taxonomy_snapshot": {
            "categories_used": [],
            "new_categories_discovered": [],
            "high_frequency_alerts": [],
        },
        "unmatched_content": {
            "v1_only": [],
            "v2_only": [],
            "note": "v0.1 uses line-level diff; semantic alignment not yet implemented",
        },
    }


def _infer_clause_ref(lines: list[str], start: int, end: int) -> str | None:
    """Try to find a clause/article number near the change block."""
    search_start = max(0, start - 3)
    search_end = min(len(lines), end + 1)
    for i in range(search_start, search_end):
        if i >= len(lines):
            break
        m = re.match(r"^([一二三四五六七八九十]+)[、，．.]", lines[i].strip())
        if m:
            return f"第{m.group(1)}条"
        m = re.match(r"^(第[一二三四五六七八九十百千\d]+[条款章节])", lines[i].strip())
        if m:
            return m.group(1)
    return None


def _summarize_change(snippet: str) -> str:
    """Generate a short title from the change snippet."""
    lines = [l.strip() for l in snippet.split("\n") if l.strip()]
    if not lines:
        return "（空内容）"
    first = lines[0]
    if len(first) > 60:
        return first[:57] + "..."
    return first


def _brief_change(change_type: str, v1_snippet: str | None, v2_snippet: str | None) -> str:
    """One-line summary for non-legal readers."""
    snippet = v1_snippet or v2_snippet or ""
    preview = snippet.replace("\n", " ")[:80].strip()
    if change_type == "added":
        return f"新增内容：{preview}..."
    elif change_type == "removed":
        return f"删除内容：{preview}..."
    else:
        return f"修改内容：{preview}..."


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [截断]"


def _calculate_coverage(v1_lines: list[str], v2_lines: list[str], opcodes) -> float:
    """Estimate what percentage of clauses were successfully aligned."""
    equal_chars = 0
    total_chars = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            equal_chars += sum(len(l) for l in v1_lines[i1:i2])
        total_chars += sum(len(l) for l in v1_lines[i1:i2])
    if total_chars == 0:
        return 0.0
    return round(equal_chars / total_chars, 4)


def run_text_diff(v1_path: str, v2_path: str) -> dict:
    """Convenience: extract + diff two contract PDFs in one call."""
    from .pdf_extractor import extract_contract

    v1 = extract_contract(v1_path)
    v2 = extract_contract(v2_path)
    return diff_contracts(v1, v2)
