#!/usr/bin/env python3
"""Reset and regenerate demo data for a repeatable showcase.

Usage:
  python scripts/prepare_demo.py              # Generate fresh demo data
  python scripts/prepare_demo.py --reset      # Wipe and regenerate
  python scripts/prepare_demo.py --only-learnings  # Only re-extract learnings
  python scripts/prepare_demo.py --seed-v2 2025 2026  # Specific V2 files

What it does:
  1. Runs the pipeline (offline) on demo PDFs if no result exists
  2. Injects realistic human verdicts (confirmed/corrected/rejected)
  3. Extracts learning records and updates the index
  4. Prints summary for verification
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Run from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

DEMO_PDFS = [
    "docs/天猫服务协议2015(2).pdf",
    "docs/天猫服务协议2026(2).pdf",
]

VERDICT_COUNT = 30  # target number of verdicts per result
CATEGORY_POOL = {
    "R01": "交付时效", "R02": "质量与鉴定标准", "R03": "付款与结算",
    "R04": "违约责任", "R07": "保险与风险承担", "R08": "运输与物流",
    "R09": "价格调整机制", "R10": "数据与隐私",
}


def run_pipeline(v1: str, v2: str, output: str) -> bool:
    """Run the pipeline in offline mode. Returns True on success."""
    import subprocess

    print(f"  运行管线: {v1} vs {v2} ...")
    result = subprocess.run(
        [sys.executable, "-m", "src.main", v1, v2, "--offline", "-o", output],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  ⚠ 管线失败:\n{result.stderr[-500:]}")
        return False
    if not os.path.exists(output):
        print(f"  ⚠ 未生成输出文件 {output}")
        return False
    print(f"  ✓ 管线完成")
    return True


def inject_verdicts(result_path: str, count: int = VERDICT_COUNT) -> dict:
    """Inject realistic human verdicts into a result JSON. Returns stats."""
    d = json.load(open(result_path))
    changes = d.get("changes", [])

    if not changes:
        print("  ⚠ 没有差异，跳过判决注入")
        return {"confirmed": 0, "corrected": 0, "rejected": 0}

    now = datetime.now(timezone.utc).isoformat()
    stats = {"confirmed": 0, "corrected": 0, "rejected": 0}

    for i, c in enumerate(changes):
        if c.get("human_verdict"):
            continue  # already has one

        cid = c.get("id", "")
        orig_level = c.get("risk_level", "medium")
        orig_cats = c.get("risk_categories", [])

        if not c.get("brief"):
            continue

        r = random.random()

        if r < 0.65 and orig_level != "low":
            # Confirm
            c["human_verdict"] = {
                "action": "confirmed",
                "timestamp": now,
                "original": {
                    "risk_level": orig_level,
                    "risk_categories": orig_cats,
                    "risk_note": c.get("risk_note", ""),
                },
            }
            stats["confirmed"] += 1

        elif r < 0.85:
            # Correct — shift level up or change category
            new_level = "high" if orig_level != "high" else "medium"
            avails = [k for k in CATEGORY_POOL if k not in orig_cats]
            new_cats = orig_cats[:]
            if avails and random.random() < 0.6:
                new_cats = [random.choice(avails)]

            c["human_verdict"] = {
                "action": "corrected",
                "corrected_risk_level": new_level,
                "corrected_risk_categories": new_cats,
                "corrected_risk_note": "人工复核后调整风险评估",
                "timestamp": now,
                "original": {
                    "risk_level": orig_level,
                    "risk_categories": orig_cats,
                    "risk_note": c.get("risk_note", ""),
                },
            }
            c["risk_level"] = new_level
            c["risk_categories"] = new_cats
            stats["corrected"] += 1

        elif r < 0.95:
            # Reject
            c["human_verdict"] = {
                "action": "rejected",
                "timestamp": now,
                "original": {
                    "risk_level": orig_level,
                    "risk_categories": orig_cats,
                    "risk_note": c.get("risk_note", ""),
                },
            }
            stats["rejected"] += 1

        total = sum(stats.values())
        if total >= count:
            break

    json.dump(d, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"  判决注入: {stats['confirmed']} 确认, {stats['corrected']} 纠偏, "
          f"{stats['rejected']} 驳回")
    return stats


def extract_and_save(result_path: str) -> None:
    """Extract learning from result and update index."""
    from src.pipeline.learning import save_learning, extract_learning

    d = json.load(open(result_path))
    job_id = Path(result_path).stem

    learning = extract_learning(d, job_id)
    save_learning(learning, Path("data/learnings"))

    hf = learning.get("human_feedback", {})
    print(f"  学习提取: {hf.get('feedback_count', 0)} 条反馈, "
          f"AI误判率 {hf.get('llm_error_rate', 0):.1%}")
    if hf.get("most_corrected_categories"):
        cats = [f"{c['name']}({c['count']}次)" for c in hf["most_corrected_categories"][:3]]
        print(f"  高频纠偏: {', '.join(cats)}")


def find_latest_result() -> str | None:
    """Find the most recent job result JSON."""
    jobs_dir = Path("data/jobs")
    if not jobs_dir.exists():
        return None
    files = sorted(jobs_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
    for f in files:
        d = json.load(open(f))
        if d.get("changes"):
            return str(f)
    return None


def reset_data():
    """Wipe learnings and job results."""
    import shutil
    for d in [Path("data/learnings"), Path("data/jobs")]:
        if d.exists():
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    print("  ✓ 演示数据已重置")


def main():
    parser = argparse.ArgumentParser(description="Prepare demo data for ContractLens showcase")
    parser.add_argument("--reset", action="store_true", help="Wipe existing data first")
    parser.add_argument("--only-learnings", action="store_true",
                        help="Only re-extract learnings from existing results")
    parser.add_argument("--seed-v2", type=int, nargs=2, metavar=("YEAR1", "YEAR2"),
                        help="Seed random generator (e.g. --seed-v2 2025 2026)")
    parser.add_argument("--count", type=int, default=VERDICT_COUNT,
                        help=f"Number of verdicts to inject (default: {VERDICT_COUNT})")
    args = parser.parse_args()

    if args.seed_v2:
        seed = args.seed_v2[0] * 10000 + args.seed_v2[1]
        random.seed(seed)
        print(f"  随机种子: {seed} (固定，确保可重复)")
    else:
        random.seed(42)
        print(f"  随机种子: 42 (默认)")

    if args.reset:
        reset_data()

    if args.only_learnings:
        latest = find_latest_result()
        if not latest:
            print("✗ 没有找到已有结果，请先运行管线")
            sys.exit(1)
        print(f"  从已有结果提取: {latest}")
        extract_and_save(latest)
        print("\n✓ 学习数据已更新")
        return

    # Run pipeline if needed
    output = "data/jobs/demo.json"
    os.makedirs("data/jobs", exist_ok=True)

    if os.path.exists(output) and not args.reset:
        print(f"  使用已有结果: {output}")
        d = json.load(open(output))
        existing_v = sum(1 for c in d.get("changes", []) if c.get("human_verdict"))
        if existing_v > 0:
            print(f"  已有 {existing_v} 条判决，跳过注入")
        else:
            inject_verdicts(output, args.count)
    else:
        if not run_pipeline(DEMO_PDFS[0], DEMO_PDFS[1], output):
            # Fallback: use existing job result
            latest = find_latest_result()
            if latest:
                print(f"  回退到已有结果: {latest}")
                output = latest
            else:
                print("✗ 管线运行失败且无已有数据")
                sys.exit(1)
        inject_verdicts(output, args.count)

    # Extract learnings
    extract_and_save(output)

    # Summary
    d = json.load(open(output))
    changes = d.get("changes", [])
    verdicts = sum(1 for c in changes if c.get("human_verdict"))
    print(f"\n{'='*50}")
    print(f"✓ 演示数据准备完成")
    print(f"  差异总数: {len(changes)}")
    print(f"  人工判决: {verdicts}")
    print(f"  结果文件: {output}")
    print(f"  学习索引: data/learnings/index.json")
    print(f"  启动演示: make web → http://localhost:8000")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
