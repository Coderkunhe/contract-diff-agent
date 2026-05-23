"""Tests for PDF extraction (Step ①a)."""
import pytest
from src.pdf_extractor import extract_contract, estimate_tokens, _strip_english, _is_cjk


class TestExtractContract:
    def test_extracts_pages(self, contract_paths):
        doc = extract_contract(contract_paths["v1"])
        assert doc.total_pages == 34

    def test_extracts_text(self, v1_doc, v2_doc):
        assert len(v1_doc.full_text) > 10000, "V1 should have substantial text"
        assert len(v2_doc.full_text) > 10000, "V2 should have substantial text"

    def test_bilingual_filtering(self, v2_doc):
        """English text should be stripped for bilingual contracts."""
        raw = v2_doc.full_text_bilingual
        filtered = v2_doc.full_text
        assert len(filtered) < len(raw), "Filtered should be shorter than bilingual"
        # Check no English-only lines remain
        for line in filtered.split("\n"):
            stripped = line.strip()
            if stripped and len(stripped) > 20:
                cjk = sum(1 for c in stripped if _is_cjk(c))
                total = len(stripped.replace(" ", "").replace(",", "").replace(".", ""))
                assert cjk / max(total, 1) > 0.1, f"Line should be Chinese: {stripped[:50]}"

    def test_v1_has_key_clauses(self, v1_doc):
        assert "协议内容及效力" in v1_doc.full_text
        assert "商户的承诺与保证" in v1_doc.full_text
        assert "违约责任" in v1_doc.full_text

    def test_v2_has_key_clauses(self, v2_doc):
        assert "协议范围及效力" in v2_doc.full_text
        assert "店铺管理及服务规范" in v2_doc.full_text
        assert "违约责任" in v2_doc.full_text

    def test_v2_is_restructured(self, v1_doc, v2_doc):
        """2026 version has different chapter structure."""
        assert "商户的承诺与保证" in v1_doc.full_text
        # V2 consolidated this into 店铺管理及服务规范
        # so the old title may not appear
        assert "店铺管理及服务规范" in v2_doc.full_text


class TestTokenEstimate:
    def test_chinese_text(self):
        text = "这是一段中文文本用于测试"
        tokens = estimate_tokens(text)
        assert tokens >= len(text) * 0.8

    def test_english_text(self):
        text = "This is English text for testing"
        tokens = estimate_tokens(text)
        assert tokens >= len(text) * 0.5

    def test_empty(self):
        assert estimate_tokens("") == 0


class TestStripEnglish:
    def test_strips_english_lines(self):
        text = "这是中文行\nThis is English\n另一行中文"
        result = _strip_english(text)
        assert "这是中文行" in result
        assert "This is English" not in result
        assert "另一行中文" in result
