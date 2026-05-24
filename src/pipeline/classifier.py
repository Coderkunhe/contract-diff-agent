"""Risk classification & friendly output — Step ⑤ of the contract diff pipeline.

Batches changes and calls LLM to assign risk categories, levels, and plain-language notes.
"""

import json
import os
from typing import Any

from json_repair import repair_json

from src.config import get_config
from src.constants.risks import RISK_CATEGORIES
from src.llm.client import AutoFallbackClient
from src.prompts.classifier import build_classifier_prompt
from .extraction import estimate_tokens


def classify_changes(
    changes: list[dict],
    api_key: str,
    model: str = "anthropic/claude-sonnet-4.6",
    base_url: str = "https://api.gmi-serving.com/v1",
    batch_size: int | None = None,
) -> list[dict]:
    cfg = get_config()
    if batch_size is None:
        batch_size = cfg.llm_batch_size
    if not changes:
        return []

    client = AutoFallbackClient(primary_model=model, timeout=cfg.llm_timeout)
    system_prompt = build_classifier_prompt()

    frequency: dict[str, int] = {c["id"]: 0 for c in RISK_CATEGORIES}
    new_categories: list[dict] = []
    classified: list[dict] = []
    total_batches = (len(changes) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(changes), batch_size):
        batch = changes[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        print(f"  分类进度: batch {batch_num}/{total_batches} "
              f"({len(batch)} 条变化)")

        changes_json = []
        for c in batch:
            changes_json.append({
                "id": c.get("id", ""),
                "change_type": c.get("change_type", "modified"),
                "brief": c.get("brief", ""),
                "clause": c.get("clause_ref_v2") or c.get("clause_ref_v1", ""),
            })

        user_prompt = f"风险分类以下合同差异:\n{json.dumps(changes_json, ensure_ascii=False)}"

        try:
            response = client.create(
                max_tokens=cfg.classify_max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                response_format={"type": "json_object"},
            )

            parts = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)

            content = "".join(parts)
            result = json.loads(repair_json(content))

            lookup = {c["id"]: c for c in result.get("classified", [])}
            for change in batch:
                cls = lookup.get(change.get("id", ""), {})
                change["risk_categories"] = cls.get("risk_categories", [])
                change["risk_level"] = cls.get("risk_level", "medium")
                change["risk_note"] = cls.get("risk_note", "")
                change["attention_for"] = cls.get("attention_for")
                change["is_favorable"] = cls.get("is_favorable")
                classified.append(change)

                for cat_id in change.get("risk_categories", []):
                    frequency[cat_id] = frequency.get(cat_id, 0) + 1

            for nc in result.get("new_categories", []):
                if nc not in new_categories:
                    new_categories.append(nc)

        except Exception as e:
            print(f"  ⚠️  分类失败 (batch {batch_num}): {e}")
            for change in batch:
                change.setdefault("risk_categories", [])
                change.setdefault("risk_level", "medium")
                change.setdefault("risk_note", "")
                classified.append(change)

    print(f"\n  分类完成: {len(classified)} 条")
    if frequency:
        top = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        cat_names = {c["id"]: c["name"] for c in RISK_CATEGORIES}
        print(f"  高频风险: {', '.join(f'{cat_names.get(k, k)}({v}次)' for k, v in top if v > 0)}")
    if new_categories:
        print(f"  新发现分类: {len(new_categories)}")

    return classified


def load_taxonomy(path: str = "data/risk_taxonomy.json") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"categories": RISK_CATEGORIES, "frequency": {}, "new_categories": []}


def save_taxonomy(frequency: dict, new_categories: list[dict],
                  path: str = "data/risk_taxonomy.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "categories": RISK_CATEGORIES,
        "frequency": frequency,
        "new_categories": new_categories,
    }
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
