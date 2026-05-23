"""Tests for validation (Step ④ L2)."""
import pytest
from src.pipeline.validator import _l2_check, _verdict_to_status, _verdict_to_confidence


class TestL2Check:
    def test_snippet_found(self):
        change = {
            "v1_snippet": "商户应在收到货物后7个工作日内完成验收",
            "v2_snippet": "商户应在收到货物后3个工作日内完成验收",
        }
        v1_text = "合同约定：商户应在收到货物后7个工作日内完成验收，逾期视为验收合格。"
        v2_text = "合同约定：商户应在收到货物后3个工作日内完成验收，逾期视为验收合格。"
        result = _l2_check(change, v1_text, v2_text)
        assert result["l2_v1_snippet_found"] is True
        assert result["l2_v2_snippet_found"] is True

    def test_snippet_not_found(self):
        change = {
            "v1_snippet": "这段话根本不存在于原文中",
            "v2_snippet": "这段也不存在",
        }
        result = _l2_check(change, "", "")
        assert result["l2_v1_snippet_found"] is False
        assert result["l2_v2_snippet_found"] is False

    def test_null_snippets_pass(self):
        """Missing snippets (None) should pass L2."""
        change = {"v1_snippet": None, "v2_snippet": None}
        result = _l2_check(change, "", "")
        assert result["l2_v1_snippet_found"] is True
        assert result["l2_v2_snippet_found"] is True

    def test_fuzzy_match(self):
        """First 30 chars matching should count as found."""
        change = {
            "v1_snippet": "这是一个很长的片段应该用模糊匹配来找" + "x" * 50,
            "v2_snippet": "短片段",
        }
        v1_text = "这是一个很长的片段应该用模糊匹配来找" + "x" * 50 + "后面还有"
        v2_text = "开头短片段结尾"
        result = _l2_check(change, v1_text, v2_text)
        assert result["l2_v1_snippet_found"] is True
        assert result["l2_v2_snippet_found"] is True

    def test_real_contract_content(self, v1_doc, v2_doc):
        """Integration: snippets from the actual contract should be found."""
        change = {
            "v1_snippet": "提前十五（15）天",
            "v2_snippet": "提前三十（30）天",
        }
        result = _l2_check(change, v1_doc.full_text, v2_doc.full_text)
        assert result["l2_v1_snippet_found"] is True
        assert result["l2_v2_snippet_found"] is True

    def test_real_contract_long_snippet(self, v1_doc, v2_doc):
        """Longer snippets work via fuzzy (first 30 chars) matching."""
        change = {
            "v1_snippet": "协议任何一方均可提前十五（15）天以书面通知的方式终止",
            "v2_snippet": "本协议任何一方均可提前三十（30）天以书面通知的方式终止",
        }
        result = _l2_check(change, v1_doc.full_text, v2_doc.full_text)
        assert result["l2_v1_snippet_found"] is True
        assert result["l2_v2_snippet_found"] is True


class TestVerdictMapping:
    def test_verdict_to_status(self):
        assert _verdict_to_status("confirmed") == "verified"
        assert _verdict_to_status("rejected") == "rejected"
        assert _verdict_to_status("uncertain") == "uncertain"

    def test_verdict_to_confidence(self):
        assert _verdict_to_confidence("confirmed") == 0.95
        assert _verdict_to_confidence("rejected") == 0.1
        assert _verdict_to_confidence("uncertain") == 0.5
