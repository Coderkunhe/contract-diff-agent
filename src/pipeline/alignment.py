"""Clause alignment — Step ② of the contract diff pipeline.

Matches clauses between V1 and V2 trees using title + content similarity.
"""

import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field

from .parsing import ClauseNode, ContractTree


@dataclass
class AlignedPair:
    v1_clause: ClauseNode | None
    v2_clause: ClauseNode | None
    similarity: float
    alignment_type: str  # "match", "restructured", "added", "removed"


@dataclass
class DiffMap:
    pairs: list[AlignedPair]
    v1_unmatched: list[ClauseNode]
    v2_unmatched: list[ClauseNode]
    v1_tree: ContractTree
    v2_tree: ContractTree


def _sim(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1, text2).ratio()


def _strip_suffixes(title: str) -> str:
    title = re.sub(r"[（(][^）)]*[）)]$", "", title).strip()
    return title


def align_clauses(tree1: ContractTree, tree2: ContractTree) -> DiffMap:
    v1_clauses = list(tree1.clauses)
    v2_clauses = list(tree2.clauses)

    scores: list[tuple[float, int, int]] = []
    for i, c1 in enumerate(v1_clauses):
        t1 = _strip_suffixes(c1.title)
        for j, c2 in enumerate(v2_clauses):
            t2 = _strip_suffixes(c2.title)
            s = _sim(t1, t2)
            if s >= 0.3:
                scores.append((s, i, j))

    scores.sort(key=lambda x: x[0], reverse=True)
    used_v1: set[int] = set()
    used_v2: set[int] = set()
    l1_pairs: list[tuple[int, int, float]] = []

    for score, i, j in scores:
        if i not in used_v1 and j not in used_v2:
            used_v1.add(i)
            used_v2.add(j)
            l1_pairs.append((i, j, score))

    l1_pairs.sort(key=lambda x: x[0])

    pairs: list[AlignedPair] = []
    v1_unmatched: list[ClauseNode] = []
    v2_unmatched: list[ClauseNode] = []

    last_v1_idx = 0
    for v1_idx, v2_idx, sim in l1_pairs:
        for k in range(last_v1_idx, v1_idx):
            if k not in used_v1:
                v1_unmatched.append(v1_clauses[k])
        last_v1_idx = v1_idx + 1

        c1 = v1_clauses[v1_idx]
        c2 = v2_clauses[v2_idx]

        l2_pairs, l2_v1_unmatched, l2_v2_unmatched = _align_l2(c1, c2)
        pairs.extend(l2_pairs)

        l1_pair = AlignedPair(
            v1_clause=c1, v2_clause=c2,
            similarity=sim, alignment_type="match",
        )
        pairs.append(l1_pair)

        for uc in l2_v1_unmatched:
            pairs.append(AlignedPair(
                v1_clause=uc, v2_clause=None,
                similarity=0.0, alignment_type="removed",
            ))
        for uc in l2_v2_unmatched:
            pairs.append(AlignedPair(
                v1_clause=None, v2_clause=uc,
                similarity=0.0, alignment_type="added",
            ))

    for k in range(last_v1_idx, len(v1_clauses)):
        if k not in used_v1:
            v1_unmatched.append(v1_clauses[k])

    for j, c in enumerate(v2_clauses):
        if j not in used_v2:
            v2_unmatched.append(c)

    for uc in v1_unmatched:
        pairs.append(AlignedPair(
            v1_clause=uc, v2_clause=None,
            similarity=0.0, alignment_type="removed",
        ))
        for child in uc.children:
            pairs.append(AlignedPair(
                v1_clause=child, v2_clause=None,
                similarity=0.0, alignment_type="removed",
            ))

    for uc in v2_unmatched:
        pairs.append(AlignedPair(
            v1_clause=None, v2_clause=uc,
            similarity=0.0, alignment_type="added",
        ))
        for child in uc.children:
            pairs.append(AlignedPair(
                v1_clause=None, v2_clause=child,
                similarity=0.0, alignment_type="added",
            ))

    return DiffMap(
        pairs=pairs, v1_unmatched=v1_unmatched, v2_unmatched=v2_unmatched,
        v1_tree=tree1, v2_tree=tree2,
    )


def _align_l2(
    parent1: ClauseNode, parent2: ClauseNode,
) -> tuple[list[AlignedPair], list[ClauseNode], list[ClauseNode]]:
    children1 = list(parent1.children)
    children2 = list(parent2.children)

    if not children1 and not children2:
        return [], [], []

    scores: list[tuple[float, int, int]] = []
    for i, c1 in enumerate(children1):
        t1 = _strip_suffixes(c1.title)
        for j, c2 in enumerate(children2):
            t2 = _strip_suffixes(c2.title)
            s = _sim(t1, t2)
            if s >= 0.25:
                scores.append((s, i, j))

    scores.sort(key=lambda x: x[0], reverse=True)
    used_v1: set[int] = set()
    used_v2: set[int] = set()
    pairs: list[AlignedPair] = []

    for score, i, j in scores:
        if i not in used_v1 and j not in used_v2:
            used_v1.add(i)
            used_v2.add(j)
            pairs.append(AlignedPair(
                v1_clause=children1[i], v2_clause=children2[j],
                similarity=score,
                alignment_type="match" if score >= 0.6 else "restructured",
            ))

    unmatched_v1 = [c for i, c in enumerate(children1) if i not in used_v1]
    unmatched_v2 = [c for i, c in enumerate(children2) if i not in used_v2]

    return sorted(pairs, key=lambda p: int(p.v1_clause.id.split(".")[-1])), unmatched_v1, unmatched_v2
