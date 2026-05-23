"""Tests for traditional_diff — the zero-LLM base layer of the pipeline."""
import pytest
from src.pipeline.traditional_diff import traditional_diff, _merge_blocks, _is_trivial, _make_brief, _truncate
from src.pipeline.parsing import ClauseNode, ContractTree
from src.pipeline.alignment import DiffMap, AlignedPair


class TestMergeBlocks:
    def test_empty(self):
        assert _merge_blocks([]) == []

    def test_single(self):
        assert _merge_blocks([("modified", "a", "b")]) == [("modified", "a", "b")]

    def test_adjacent_same_type_merged(self):
        blocks = [
            ("modified", "foo", "bar"),
            ("modified", "baz", "qux"),
        ]
        result = _merge_blocks(blocks)
        assert len(result) == 1
        assert result[0][0] == "modified"
        assert "foo" in result[0][1] and "baz" in result[0][1]
        assert "bar" in result[0][2] and "qux" in result[0][2]

    def test_different_types_not_merged(self):
        blocks = [
            ("added", "", "new"),
            ("removed", "old", ""),
        ]
        result = _merge_blocks(blocks)
        assert len(result) == 2


class TestIsTrivial:
    def test_empty_strings(self):
        assert _is_trivial(("modified", "", "")) is True

    def test_whitespace_only_diff(self):
        assert _is_trivial(("modified", "hello world", "hello  world")) is True

    def test_punctuation_only_diff(self):
        # Punctuation differs but meaningful chars are the same → whitespace
        # normalization doesn't catch this, so it's not considered trivial
        assert _is_trivial(("modified", "hello.", "hello。")) is False

    def test_meaningful_diff(self):
        assert _is_trivial(("modified", "赔付500元", "赔付1000元")) is False

    def test_very_short_meaningful(self):
        # Two meaningful CJK characters → non-trivial
        assert _is_trivial(("modified", "一", "二")) is False

    def test_single_char(self):
        # Two single chars → 2 meaningful chars, threshold is < 2, so passes
        assert _is_trivial(("modified", "x", "y")) is False


class TestMakeBrief:
    def test_added(self):
        brief = _make_brief("added", "", "新条款内容在这里需要更多文字来展示截断效果", "第一条 测试")
        assert "新增" in brief
        assert "第一条 测试" in brief

    def test_removed(self):
        brief = _make_brief("removed", "旧条款内容被删除", "", "第二条 旧条款")
        assert "删除" in brief

    def test_modified(self):
        brief = _make_brief("modified", "旧文本", "新文本", "第三条 变更")
        assert "修改" in brief
        assert "旧文本" in brief
        assert "新文本" in brief


class TestTruncate:
    def test_short_text(self):
        assert _truncate("hello", 10) == "hello"

    def test_long_text(self):
        result = _truncate("x" * 1000, 500)
        assert len(result) <= 510  # 500 + truncation suffix
        assert "截断" in result


class TestTraditionalDiff:
    def _make_clause(self, number="第一条", title="测试条款", full_text="测试内容", idx=1):
        return ClauseNode(
            id=f"clause-{idx}", level=1, number=number, title=title,
            full_text=full_text, page_start=None, children=[],
        )

    def _make_tree(self, clauses):
        return ContractTree(
            file_path="test.pdf", total_pages=1, clauses=clauses,
            attachments=[], tables=[], full_text_sections={},
        )

    def _make_diff_map(self, pairs):
        tree = self._make_tree([])
        return DiffMap(pairs=pairs, v1_unmatched=[], v2_unmatched=[], v1_tree=tree, v2_tree=tree)

    def test_empty_pairs(self):
        dm = self._make_diff_map([])
        changes = traditional_diff(dm)
        assert changes == []

    def test_added_clause(self):
        c2 = self._make_clause("第二条", "新增条款", "这是新加的条款内容")
        pair = AlignedPair(v1_clause=None, v2_clause=c2, similarity=0.0, alignment_type="added")
        dm = self._make_diff_map([pair])
        changes = traditional_diff(dm)
        assert len(changes) == 1
        assert changes[0]["change_type"] == "added"
        assert changes[0]["clause_ref_v1"] is None
        assert changes[0]["clause_ref_v2"] == "第二条、新增条款"
        assert changes[0]["source"] == "algorithm"

    def test_removed_clause(self):
        c1 = self._make_clause("第一条", "删除条款", "这个条款被删了")
        pair = AlignedPair(v1_clause=c1, v2_clause=None, similarity=0.0, alignment_type="removed")
        dm = self._make_diff_map([pair])
        changes = traditional_diff(dm)
        assert len(changes) == 1
        assert changes[0]["change_type"] == "removed"
        assert changes[0]["clause_ref_v2"] is None
        assert changes[0]["v2_snippet"] is None

    def test_identical_matched_pair(self):
        text = "完全相同的条款文本内容"
        c1 = self._make_clause("第一条", "相同条款", text)
        c2 = self._make_clause("第一条", "相同条款", text)
        pair = AlignedPair(v1_clause=c1, v2_clause=c2, similarity=1.0, alignment_type="match")
        dm = self._make_diff_map([pair])
        changes = traditional_diff(dm)
        assert len(changes) == 0  # No changes for identical text

    def test_modified_clause(self):
        c1 = self._make_clause("第一条", "变更条款",
            "商户应在收到货物后7个工作日内完成验收")
        c2 = self._make_clause("第一条", "变更条款",
            "商户应在收到货物后3个工作日内完成验收")
        pair = AlignedPair(v1_clause=c1, v2_clause=c2, similarity=0.9, alignment_type="match")
        dm = self._make_diff_map([pair])
        changes = traditional_diff(dm)
        assert len(changes) >= 1
        # Should produce at least one modified change
        modified = [c for c in changes if c["change_type"] == "modified"]
        assert len(modified) >= 1
        assert "7" in modified[0]["v1_snippet"] or "3" in modified[0]["v2_snippet"]

    def test_output_schema(self):
        c1 = self._make_clause("第一条", "条款", "旧文本ABC旧")
        c2 = self._make_clause("第一条", "条款", "新文本XYZ新")
        pair = AlignedPair(v1_clause=c1, v2_clause=c2, similarity=0.8, alignment_type="match")
        dm = self._make_diff_map([pair])
        changes = traditional_diff(dm)
        for c in changes:
            assert "id" in c
            assert c["id"].startswith("diff-")
            assert "change_type" in c and c["change_type"] in ("added", "removed", "modified")
            assert "clause_ref_v1" in c
            assert "clause_ref_v2" in c
            assert "v1_snippet" in c
            assert "v2_snippet" in c
            assert "brief" in c
            assert c.get("source") == "algorithm"
            assert c.get("human_note") is None
            assert c.get("risk_categories") == []
            assert c.get("risk_level") == "low"

    def test_offline_pipeline_no_llm_dependency(self, diff_map):
        """Integration: traditional_diff runs on real aligned contracts without any LLM."""
        # No API key, no model — this should just work
        changes = traditional_diff(diff_map)
        assert isinstance(changes, list)
        assert len(changes) > 0, "Should find changes in real contract comparison"
        # All changes should be algorithm-sourced
        for c in changes:
            assert c.get("source") == "algorithm"
            assert "id" in c
            assert "change_type" in c

    def test_all_change_types_in_real_data(self, diff_map):
        """Real contract diff should produce a mix of change types."""
        changes = traditional_diff(diff_map)
        types = {c["change_type"] for c in changes}
        # Should have at least modified changes; added/removed depend on alignment
        assert "modified" in types

    def test_empty_clause_texts_produce_no_changes(self):
        c1 = self._make_clause("第一条", "空条款", "")
        c2 = self._make_clause("第一条", "空条款", "")
        pair = AlignedPair(v1_clause=c1, v2_clause=c2, similarity=1.0, alignment_type="match")
        dm = self._make_diff_map([pair])
        changes = traditional_diff(dm)
        assert len(changes) == 0
