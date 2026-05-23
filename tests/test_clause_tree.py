"""Tests for clause tree extraction (Step ①b)."""
import pytest
from src.pipeline.parsing import build_clause_tree, ClauseNode


class TestBuildClauseTree:
    def test_extracts_l1_chapters(self, v1_tree, v2_tree):
        assert len(v1_tree.clauses) >= 16, "V1 should have at least 16 L1 chapters"
        assert len(v2_tree.clauses) >= 14, "V2 should have at least 14 L1 chapters"

    def test_extracts_l2_subclauses(self, v1_tree, v2_tree):
        v1_l2 = sum(len(c.children) for c in v1_tree.clauses)
        v2_l2 = sum(len(c.children) for c in v2_tree.clauses)
        assert v1_l2 >= 40, f"V1 has {v1_l2} L2 sub-clauses"
        assert v2_l2 >= 30, f"V2 has {v2_l2} L2 sub-clauses"

    def test_key_chapters_present(self, v1_tree):
        titles = {c.title for c in v1_tree.clauses}
        assert "协议内容及效力" in titles
        assert "定义" in titles
        assert "违约责任" in titles
        assert "协议的终止" in titles

    def test_2026_bilingual_titles(self, v2_tree):
        """2026 contract has bilingual chapter titles."""
        bilingual = [c for c in v2_tree.clauses if "（" in c.title and "）" in c.title]
        assert len(bilingual) >= 10, f"Should have bilingual titles, found {len(bilingual)}"

    def test_l2_clauses_have_ids(self, v1_tree):
        for c in v1_tree.clauses:
            for child in c.children:
                assert child.id.startswith(f"{c.id}."), \
                    f"Child {child.id} should start with parent {c.id}"

    def test_no_empty_key_chapters(self, v1_tree):
        """Key chapters should have text content."""
        empty = [c for c in v1_tree.clauses
                 if not c.full_text and not c.children
                 and len(c.title) > 10]
        assert len(empty) == 0, f"Empty chapters: {[c.title for c in empty]}"

    def test_text_preserved(self, v1_tree):
        """Extracted text should be substantial."""
        total = sum(len(c.full_text) for c in v1_tree.clauses)
        total += sum(len(child.full_text) for c in v1_tree.clauses for child in c.children)
        assert total > 5000, f"Total clause text: {total} chars"

    def test_no_table_artifacts(self, v1_tree):
        """Table garbled text should not appear as clause titles."""
        for c in v1_tree.clauses:
            assert "三" not in c.title.split("、")[0][:1] or "协议" in c.title or \
                "商户" in c.title or "定义" in c.title or "保证金" in c.title or \
                "服务" in c.title or "积分" in c.title or "消费者" in c.title or \
                "承诺" in c.title or "天猫" in c.title or "保密" in c.title or \
                "责任" in c.title or "终止" in c.title or "通知" in c.title or \
                "争议" in c.title or "附件" in c.title or "商业" in c.title, \
                f"Suspicious table artifact: {c.title}"
