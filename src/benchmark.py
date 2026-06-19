from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for e in expected if e.lower() in answer.lower())
    if hits == 0:
        return 0.0
    if hits == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    base = recall_points(answer, expected)
    bonus = 0.1 if len(answer.strip()) > 20 else 0.0
    return min(1.0, base + bonus)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    total_compactions = 0

    for conv in conversations:
        user_id: str = conv["user_id"]
        train_thread = f"{conv['id']}-train"

        for turn in conv["turns"]:
            result = agent.reply(user_id, train_thread, turn)
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        # Ask recall questions in a FRESH thread (cross-session)
        for idx, rq in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{conv['id']}-recall-{idx}"
            result = agent.reply(user_id, recall_thread, rq["question"])
            answer = result.get("response", "")
            expected = rq.get("expected_contains", [])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        total_compactions += agent.compaction_count(train_thread)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # Memory growth: sum of all User.md sizes (advanced) or 0 (baseline)
    memory_bytes = 0
    user_ids = {conv["user_id"] for conv in conversations}
    if hasattr(agent, "memory_file_size"):
        for uid in user_ids:
            memory_bytes += agent.memory_file_size(uid)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_bytes,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    try:
        from tabulate import tabulate
        headers = [
            "Agent",
            "Agent tokens only",
            "Prompt tokens processed",
            "Cross-session recall",
            "Response quality",
            "Memory growth (bytes)",
            "Compactions",
        ]
        data = [
            [
                r.agent_name,
                r.agent_tokens_only,
                r.prompt_tokens_processed,
                f"{r.recall_score:.3f}",
                f"{r.response_quality:.3f}",
                r.memory_growth_bytes,
                r.compactions,
            ]
            for r in rows
        ]
        return tabulate(data, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback plain text
        lines = ["| " + " | ".join(["Agent", "Agent tokens only", "Prompt tokens processed",
                                     "Cross-session recall", "Response quality",
                                     "Memory growth (bytes)", "Compactions"]) + " |"]
        lines.append("|" + "|".join(["---"] * 7) + "|")
        for r in rows:
            lines.append(
                f"| {r.agent_name} | {r.agent_tokens_only} | {r.prompt_tokens_processed} |"
                f" {r.recall_score:.3f} | {r.response_quality:.3f} |"
                f" {r.memory_growth_bytes} | {r.compactions} |"
            )
        return "\n".join(lines)


def _print_analysis(label: str, rows: list[BenchmarkRow]) -> None:
    if len(rows) < 2:
        return
    baseline = next((r for r in rows if "Baseline" in r.agent_name), rows[0])
    advanced = next((r for r in rows if "Advanced" in r.agent_name), rows[1])

    recall_diff = advanced.recall_score - baseline.recall_score
    prompt_diff_pct = (
        (advanced.prompt_tokens_processed - baseline.prompt_tokens_processed)
        / max(1, baseline.prompt_tokens_processed) * 100
    )
    print(f"\n[{label}] Advanced recall: {advanced.recall_score:.3f} vs "
          f"Baseline: {baseline.recall_score:.3f} (Δ {recall_diff:+.3f})")
    print(f"[{label}] Advanced prompt tokens: {advanced.prompt_tokens_processed} | "
          f"Baseline: {baseline.prompt_tokens_processed} "
          f"({prompt_diff_pct:+.1f}% vs Baseline)")
    if advanced.compactions > 0:
        print(f"[{label}] Advanced compactions: {advanced.compactions} "
              f"(compact memory kích hoạt để giữ prompt cost ổn định)")
    if label.startswith("Stress") and advanced.compactions > 0:
        reduction = (
            (baseline.prompt_tokens_processed - advanced.prompt_tokens_processed)
            / max(1, baseline.prompt_tokens_processed) * 100
        )
        print(f"[{label}] Compact memory giảm prompt tokens "
              f"{reduction:.1f}% so với Baseline ở hội thoại dài")


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    offline = not bool(config.model.api_key)
    mode_label = "offline (deterministic)" if offline else f"live ({config.model.provider}/{config.model.model_name})"
    print(f"\nMode: {mode_label}")

    # ── Standard Benchmark ────────────────────────────────────────────────────
    std_path = config.data_dir / "conversations.json"
    std_convs = load_conversations(std_path)
    print(f"\n{'='*60}")
    print("STANDARD BENCHMARK  (data/conversations.json)")
    print(f"{'='*60}")

    baseline_std = BaselineAgent(config=config, force_offline=offline)
    advanced_std = AdvancedAgent(config=config, force_offline=offline)

    std_rows = [
        run_agent_benchmark("Baseline", baseline_std, std_convs, config),
        run_agent_benchmark("Advanced", advanced_std, std_convs, config),
    ]
    print(format_rows(std_rows))
    _print_analysis("Standard", std_rows)

    # ── Long-Context Stress Benchmark ─────────────────────────────────────────
    stress_path = config.data_dir / "advanced_long_context.json"
    stress_convs = load_conversations(stress_path)
    print(f"\n{'='*60}")
    print("LONG-CONTEXT STRESS BENCHMARK  (data/advanced_long_context.json)")
    print(f"{'='*60}")

    baseline_stress = BaselineAgent(config=config, force_offline=offline)
    advanced_stress = AdvancedAgent(config=config, force_offline=offline)

    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, config),
    ]
    print(format_rows(stress_rows))
    _print_analysis("Stress", stress_rows)

    print()


if __name__ == "__main__":
    main()
