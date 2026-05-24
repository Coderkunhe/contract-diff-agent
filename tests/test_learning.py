"""Tests for self-evolution learning module."""

import json
import pytest
from pathlib import Path

from src.pipeline.learning import (
    extract_learning,
    save_learning,
    load_past_learnings,
    format_learnings_context,
    _load_index,
    _update_index,
    _resolve_dir,
)


# ── Sample fixtures ────────────────────────────────────────────────


def _make_result(**overrides):
    """Build a minimal pipeline result dict for testing."""
    base = {
        "meta": {
            "contract_v1": "v1.pdf",
            "contract_v2": "v2.pdf",
            "compared_at": "2026-05-24T12:00:00.000+00:00",
            "agent_version": "1.4.0",
            "pipeline": "v04-llm_enhanced",
            "model": "deepseek-chat",
            "token_estimate": {"v1_tokens": 5000, "v2_tokens": 5200, "total": 10200},
        },
        "diff_summary": {
            "total_changes": 3,
            "confirmed": 2,
            "high_risk": 1,
            "medium_risk": 1,
            "low_risk": 1,
            "alignment_coverage": 0.95,
            "llm_enhanced": True,
        },
        "changes": [
            {
                "id": "diff-001",
                "change_type": "modified",
                "brief": "验收期限从7个工作日缩短为3个工作日",
                "risk_categories": ["R01"],
                "risk_level": "high",
                "risk_note": "验收时间缩短可能导致来不及充分检查",
                "validation": {"l3_verdict": "confirmed", "confidence": 0.95, "status": "verified"},
                "human_note": None,
            },
            {
                "id": "diff-002",
                "change_type": "added",
                "brief": "新增保证金条款",
                "risk_categories": ["R03"],
                "risk_level": "medium",
                "risk_note": "新增保证金要求",
                "validation": {"l3_verdict": "confirmed", "confidence": 0.88, "status": "verified"},
                "human_note": "采购部已确认",
            },
            {
                "id": "diff-003",
                "change_type": "removed",
                "brief": "删除知识产权条款",
                "risk_categories": ["R05"],
                "risk_level": "high",
                "risk_note": "知识产权条款被删除",
                "validation": {"l3_verdict": "rejected", "confidence": 0.45, "status": "rejected"},
                "human_note": "",
            },
        ],
        "risk_taxonomy_snapshot": {
            "categories_used": ["R01", "R03", "R05"],
            "frequency": {"交付时效": 1, "付款与结算": 1, "知识产权与品牌": 1},
        },
        "unmatched_content": {"v1_only": [], "v2_only": []},
    }
    base.update(overrides)
    return base


# ── Test extract_learning ──────────────────────────────────────────


class TestExtractLearning:
    def test_extracts_risk_profile(self):
        result = _make_result()
        learning = extract_learning(result, "test001")

        assert learning["job_id"] == "test001"
        rp = learning["risk_profile"]
        assert rp["total_changes"] == 3
        assert rp["high_risk_count"] == 1
        assert rp["medium_risk_count"] == 1
        assert rp["low_risk_count"] == 1
        assert len(rp["top_categories"]) == 3

    def test_extracts_quality_signals(self):
        result = _make_result()
        learning = extract_learning(result, "test001")

        qs = learning["quality_signals"]
        assert qs["validation_rejection_rate"] > 0
        assert qs["human_corrections_count"] == 1
        assert qs["high_confidence_verified_count"] == 1  # diff-001 has conf 0.95, status verified

    def test_extracts_high_confidence_patterns(self):
        result = _make_result()
        # Add a duplicate high-confidence pattern
        result["changes"].append({
            "id": "diff-004",
            "change_type": "modified",
            "brief": "验收周期调整",
            "risk_categories": ["R01"],
            "risk_level": "high",
            "validation": {"l3_verdict": "confirmed", "confidence": 0.97, "status": "verified"},
            "human_note": None,
        })
        learning = extract_learning(result, "test001")

        patterns = learning["high_confidence_patterns"]
        found_r01_high = [p for p in patterns if p["category"] == "R01" and p["risk_level"] == "high"]
        assert len(found_r01_high) >= 1
        assert found_r01_high[0]["count"] >= 2

    def test_handles_empty_changes(self):
        result = _make_result()
        result["changes"] = []
        result["diff_summary"] = {"total_changes": 0, "confirmed": 0, "high_risk": 0, "medium_risk": 0, "low_risk": 0, "alignment_coverage": 1, "llm_enhanced": False}
        result["risk_taxonomy_snapshot"] = {}
        learning = extract_learning(result, "empty001")

        assert learning["risk_profile"]["total_changes"] == 0
        assert learning["quality_signals"]["validation_rejection_rate"] == 0
        assert learning["quality_signals"]["human_corrections_count"] == 0

    def test_extracts_human_notes(self):
        result = _make_result()
        learning = extract_learning(result, "note001")

        assert learning["quality_signals"]["human_corrections_count"] == 1
        assert len(learning["human_notes"]) == 1
        assert learning["human_notes"][0]["change_id"] == "diff-002"

    def test_builds_summary_string(self):
        result = _make_result()
        learning = extract_learning(result, "sum001")

        s = learning["summary"]
        assert "3条差异" in s
        assert "v04-llm_enhanced" in s

    def test_handles_offline_mode(self):
        result = _make_result()
        result["meta"]["model"] = None
        result["meta"]["pipeline"] = "v04-offline"
        result["diff_summary"]["llm_enhanced"] = False
        learning = extract_learning(result, "off001")

        assert learning["meta"]["model"] == "offline"
        assert "offline" in learning["summary"]

    def test_handles_missing_validation_field(self):
        result = _make_result()
        for c in result["changes"]:
            c.pop("validation", None)
        learning = extract_learning(result, "noval001")

        assert learning["quality_signals"]["validation_rejection_rate"] == 0
        assert learning["quality_signals"]["high_confidence_verified_count"] == 0


# ── Test save / load ───────────────────────────────────────────────


class TestSaveAndLoad:
    def test_save_creates_files(self, tmp_path):
        result = _make_result()
        learning = extract_learning(result, "save001")
        path = save_learning(learning, data_dir=tmp_path)

        assert path.exists()
        assert path.suffix == ".json"
        assert (tmp_path / "index.json").exists()

    def test_index_upserts(self, tmp_path):
        result = _make_result()
        for jid in ["run-a", "run-b"]:
            learning = extract_learning(result, jid)
            save_learning(learning, data_dir=tmp_path)

        # Re-save run-a with different data
        result2 = _make_result()
        result2["diff_summary"]["total_changes"] = 99
        learning2 = extract_learning(result2, "run-a")
        save_learning(learning2, data_dir=tmp_path)

        index = _load_index(tmp_path)
        assert index["total_runs"] == 2
        run_a = [r for r in index["runs"] if r["job_id"] == "run-a"][0]
        assert run_a["total_changes"] == 99

    def test_load_returns_recent(self, tmp_path):
        result = _make_result()
        for jid in ["run-a", "run-b", "run-c"]:
            learning = extract_learning(result, jid)
            save_learning(learning, data_dir=tmp_path)

        past = load_past_learnings(limit=2, data_dir=tmp_path)
        assert len(past) == 2

    def test_load_empty_dir(self, tmp_path):
        past = load_past_learnings(data_dir=tmp_path)
        assert past == []

    def test_load_corrupt_index(self, tmp_path):
        (tmp_path / "index.json").write_text("{not valid json")
        past = load_past_learnings(data_dir=tmp_path)
        assert past == []


# ── Test format_learnings_context ──────────────────────────────────


class TestFormatContext:
    def test_empty_learnings(self):
        assert format_learnings_context([]) == ""

    def test_formats_learning_list(self):
        learnings = [{
            "job_id": "a1", "timestamp": "2026-05-24T12:00:00Z",
            "pipeline": "v04-llm_enhanced", "model": "deepseek-chat",
            "total_changes": 100, "high_risk": 5, "medium_risk": 20, "low_risk": 75,
            "top_category": {"id": "R01", "name": "交付时效", "count": 15},
            "top_categories": [{"id": "R01", "name": "交付时效", "count": 15}],
            "validation_rejection_rate": 0.03,
            "human_corrections_count": 2,
            "summary": "...",
        }]
        ctx = format_learnings_context(learnings)

        assert "历史比对经验" in ctx
        assert "交付时效" in ctx
        assert "5条" in ctx  # high_risk
        assert "LLM校验拒绝率" in ctx

    def test_formats_multiple_learnings(self):
        learnings = [
            {
                "job_id": "a1", "timestamp": "2026-05-24T12:00:00Z",
                "pipeline": "v04-llm_enhanced", "model": "deepseek-chat",
                "total_changes": 100, "high_risk": 5, "medium_risk": 20, "low_risk": 75,
                "top_category": {"id": "R01", "name": "交付时效", "count": 15},
                "top_categories": [{"id": "R01", "name": "交付时效", "count": 15}],
                "validation_rejection_rate": 0.03,
                "human_corrections_count": 2,
                "summary": "...",
            },
            {
                "job_id": "b2", "timestamp": "2026-05-23T12:00:00Z",
                "pipeline": "v04-llm_enhanced", "model": "deepseek-chat",
                "total_changes": 80, "high_risk": 3, "medium_risk": 15, "low_risk": 62,
                "top_category": {"id": "R03", "name": "付款与结算", "count": 20},
                "top_categories": [{"id": "R03", "name": "付款与结算", "count": 20}],
                "validation_rejection_rate": 0.05,
                "human_corrections_count": 1,
                "summary": "...",
            },
        ]
        ctx = format_learnings_context(learnings)

        assert "2 次过往合同比对" in ctx
        assert "重点关注" in ctx

    def test_none_rejection_rate(self):
        learnings = [{
            "job_id": "a1", "timestamp": "2026-05-24T12:00:00Z",
            "pipeline": "v04-offline", "model": "offline",
            "total_changes": 100, "high_risk": 5, "medium_risk": 20, "low_risk": 75,
            "top_category": None,
            "top_categories": [],
            "validation_rejection_rate": None,
            "human_corrections_count": 0,
            "summary": "...",
        }]
        ctx = format_learnings_context(learnings)
        assert "N/A" in ctx


# ── Update_index ───────────────────────────────────────────────────


class TestUpdateIndex:
    def test_creates_global_trends(self, tmp_path):
        result = _make_result()
        learning = extract_learning(result, "t001")
        index = _update_index(learning, tmp_path)

        assert index["total_runs"] == 1
        trends = index["global_trends"]
        assert trends["avg_total_changes"] == 3
        assert trends["avg_high_risk_count"] == 1
        assert trends["total_human_corrections"] == 1

    def test_aggregates_multiple_runs(self, tmp_path):
        result = _make_result()
        for jid in ["t001", "t002", "t003"]:
            learning = extract_learning(result, jid)
            save_learning(learning, data_dir=tmp_path)

        # Save a 4th — _update_index runs inside save_learning, reads existing index from disk
        learning4 = extract_learning(result, "t004")
        save_learning(learning4, data_dir=tmp_path)

        index = _load_index(tmp_path)
        assert index["total_runs"] == 4
        assert index["global_trends"]["total_human_corrections"] == 4
