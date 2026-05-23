"""Tests for clause alignment (Step ②)."""
import pytest
from src.clause_aligner import align_clauses, _sim, _strip_suffixes


class TestSimilarity:
    def test_identical(self):
        assert _sim("违约责任", "违约责任") > 0.9

    def test_similar(self):
        """Titles with shared characters should have decent similarity."""
        assert _sim("协议内容及效力", "协议范围及效力") > 0.5

    def test_different(self):
        assert _sim("违约责任", "协议范围及效力") < 0.4


class TestStripSuffixes:
    def test_strips_english(self):
        assert _strip_suffixes("定义（Definition）") == "定义"

    def test_preserves_plain(self):
        assert _strip_suffixes("违约责任") == "违约责任"


class TestAlignClauses:
    def test_produces_matches(self, diff_map):
        matched = [p for p in diff_map.pairs
                   if p.alignment_type == "match" and p.v1_clause.level == 1]
        assert len(matched) >= 10, f"Should have at least 10 L1 matches, got {len(matched)}"

    def test_matched_clauses_are_consistent(self, diff_map):
        """Matched pairs should have both sides present."""
        for p in diff_map.pairs:
            if p.alignment_type == "match":
                assert p.v1_clause is not None
                assert p.v2_clause is not None
                assert p.similarity >= 0.3

    def test_correct_key_match(self, diff_map):
        """Known semantic equivalents should be matched."""
        for p in diff_map.pairs:
            if p.v1_clause and p.v1_clause.level == 1:
                if p.v1_clause.title == "违约责任" and p.v2_clause:
                    assert "违约责任" in p.v2_clause.title

    def test_chapter_numbering_shift_caught(self, diff_map):
        """V1's '一、协议内容及效力' should match V2's chapter about scope."""
        found = False
        for p in diff_map.pairs:
            if (p.v1_clause and p.v1_clause.title == "协议内容及效力"
                    and p.alignment_type == "match"):
                found = True
                break
        assert found, "Chapter 1 should be matched despite renumbering"

    def test_unmatched_are_valid(self, diff_map):
        """Unmatched clauses should exist in their respective trees."""
        for c in diff_map.v1_unmatched:
            assert c in diff_map.v1_tree.clauses
        for c in diff_map.v2_unmatched:
            assert c in diff_map.v2_tree.clauses
