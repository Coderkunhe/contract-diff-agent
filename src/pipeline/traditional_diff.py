"""Clause-level traditional diff — zero LLM dependency.

Takes aligned clause pairs from the alignment step and uses SequenceMatcher
to find per-clause additions, deletions, and modifications.

This is the base layer of the pipeline — always runs, works offline,
produces a fully usable report even when LLM is unavailable.
"""

from difflib import SequenceMatcher

from .alignment import DiffMap


def traditional_diff(diff_map: DiffMap) -> list[dict]:
    """Run clause-level text diff on all aligned pairs.

    Returns a list of change dicts with:
      id, change_type, clause_ref_v1, clause_ref_v2,
      v1_snippet, v2_snippet, brief, source="algorithm",
      human_note=None
    """
    changes: list[dict] = []
    counter = 0

    for pair in diff_map.pairs:
        if pair.alignment_type == "added":
            c2 = pair.v2_clause
            counter += 1
            changes.append({
                "id": f"diff-{counter:03d}",
                "change_type": "added",
                "clause_ref_v1": None,
                "clause_ref_v2": f"{c2.number}、{c2.title}" if c2 else None,
                "v1_snippet": None,
                "v2_snippet": _truncate(c2.full_text, 500) if c2 else None,
                "brief": f"新增条款：{c2.title}" if c2 and c2.title else "新增内容",
                "source": "algorithm",
                "human_note": None,
            })
            continue

        if pair.alignment_type == "removed":
            c1 = pair.v1_clause
            counter += 1
            changes.append({
                "id": f"diff-{counter:03d}",
                "change_type": "removed",
                "clause_ref_v1": f"{c1.number}、{c1.title}" if c1 else None,
                "clause_ref_v2": None,
                "v1_snippet": _truncate(c1.full_text, 500) if c1 else None,
                "v2_snippet": None,
                "brief": f"删除条款：{c1.title}" if c1 and c1.title else "删除内容",
                "source": "algorithm",
                "human_note": None,
            })
            continue

        # Matched or restructured pair — diff the text
        c1 = pair.v1_clause
        c2 = pair.v2_clause
        if not c1 or not c2:
            continue

        t1 = c1.full_text.strip()
        t2 = c2.full_text.strip()

        if not t1 and not t2:
            continue
        if t1 == t2:
            continue  # Identical — no changes

        clause_ref_v1 = f"{c1.number}、{c1.title}"
        clause_ref_v2 = f"{c2.number}、{c2.title}"

        # Run SequenceMatcher on the clause text
        matcher = SequenceMatcher(None, t1, t2)
        blocks = matcher.get_matching_blocks()

        # Collect non-matching regions as changes
        # We merge adjacent small gaps for readability
        v1_pos = 0
        v2_pos = 0
        changed_blocks: list[tuple[str, str, str]] = []  # (type, v1_text, v2_text)

        for block in blocks:
            gap1_start = v1_pos
            gap1_end = block.a
            gap2_start = v2_pos
            gap2_end = block.b

            has_gap1 = gap1_end > gap1_start
            has_gap2 = gap2_end > gap2_start

            if has_gap1 and has_gap2:
                # Both sides changed → modified
                changed_blocks.append((
                    "modified",
                    t1[gap1_start:gap1_end].strip(),
                    t2[gap2_start:gap2_end].strip(),
                ))
            elif has_gap1 and not has_gap2:
                changed_blocks.append((
                    "removed",
                    t1[gap1_start:gap1_end].strip(),
                    "",
                ))
            elif not has_gap1 and has_gap2:
                changed_blocks.append((
                    "added",
                    "",
                    t2[gap2_start:gap2_end].strip(),
                ))

            v1_pos = block.a + block.size
            v2_pos = block.b + block.size

        # Merge adjacent same-type blocks
        merged = _merge_blocks(changed_blocks)

        # Skip trivial whitespace-only changes
        merged = [b for b in merged if not _is_trivial(b)]

        for ctype, s1, s2 in merged:
            counter += 1
            brief = _make_brief(ctype, s1, s2, clause_ref_v2)

            # Heuristic risk level when no LLM available
            if ctype == "removed":
                est_risk = "high"
            elif ctype == "added":
                est_risk = "medium"
            else:
                est_risk = "medium"

            changes.append({
                "id": f"diff-{counter:03d}",
                "change_type": ctype,
                "clause_ref_v1": clause_ref_v1,
                "clause_ref_v2": clause_ref_v2,
                "v1_snippet": _truncate(s1, 500) if s1 else None,
                "v2_snippet": _truncate(s2, 500) if s2 else None,
                "brief": brief,
                "risk_categories": [],
                "risk_level": est_risk,
                "risk_note": "风险等级由传统算法估算，未经LLM校验，请人工确认",
                "attention_for": None,
                "is_favorable": None,
                "source": "algorithm",
                "human_note": None,
            })

    return changes


def _merge_blocks(blocks: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Merge adjacent same-type blocks for readability."""
    if len(blocks) <= 1:
        return blocks

    merged = []
    current_type, s1, s2 = blocks[0]
    for ctype, t1, t2 in blocks[1:]:
        if ctype == current_type:
            s1 = (s1 + "\n" + t1).strip() if t1 else s1
            s2 = (s2 + "\n" + t2).strip() if t2 else s2
        else:
            merged.append((current_type, s1, s2))
            current_type, s1, s2 = ctype, t1, t2
    merged.append((current_type, s1, s2))
    return merged


def _is_trivial(block: tuple[str, str, str]) -> bool:
    """Filter out whitespace/punctuation-only changes."""
    _, s1, s2 = block
    combined = (s1 + s2).strip()
    if not combined:
        return True
    # Skip if the only difference is whitespace
    if s1.replace(" ", "").replace("\n", "") == s2.replace(" ", "").replace("\n", ""):
        return True
    # Skip very short non-semantic changes (< 3 meaningful chars)
    meaningful = [c for c in combined if c.isalnum() or '一' <= c <= '鿿']
    if len(meaningful) < 2:
        return True
    return False


def _make_brief(change_type: str, s1: str, s2: str, clause_ref: str) -> str:
    """Generate a short algorithmic brief."""
    prefix = {"added": "新增", "removed": "删除", "modified": "修改"}.get(change_type, "变更")
    location = f"「{clause_ref}」" if clause_ref else ""

    if change_type == "added":
        preview = s2.replace("\n", " ")[:50]
        return f"{prefix}：{location}{preview}..."
    elif change_type == "removed":
        preview = s1.replace("\n", " ")[:50]
        return f"{prefix}：{location}{preview}..."
    else:
        # Show both old and new briefly
        old = s1.replace("\n", " ")[:25]
        new = s2.replace("\n", " ")[:25]
        return f"{prefix}：{location}「{old}」→「{new}」"


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [截断]"
