from __future__ import annotations

from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        model=model_cfg,
        judge_model=model_cfg,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    # Initial read returns default content
    default = store.read_text("u1")
    assert "User Profile" in default
    assert store.file_size("u1") == 0  # file does not exist yet

    # Write
    store.write_text("u1", "# Profile\nname: Alice\n")
    assert "Alice" in store.read_text("u1")
    assert store.file_size("u1") > 0

    # Edit (replacement)
    changed = store.edit_text("u1", "Alice", "Bob")
    assert changed is True
    assert "Bob" in store.read_text("u1")
    assert "Alice" not in store.read_text("u1")

    # Edit with non-existing text
    not_changed = store.edit_text("u1", "NonExistent", "X")
    assert not_changed is False

    # upsert_fact: add new key
    store.upsert_fact("u1", "city", "Hanoi")
    assert "city: Hanoi" in store.read_text("u1")

    # upsert_fact: update existing key (no duplicates)
    store.upsert_fact("u1", "city", "Saigon")
    text = store.read_text("u1")
    assert "city: Saigon" in text
    assert text.count("city:") == 1  # no duplicate


def test_compact_trigger(tmp_path: Path) -> None:
    mgr = CompactMemoryManager(threshold_tokens=50, keep_messages=2)

    # 10 messages * 30 chars ≈ 7 tokens each = 70 total → triggers compact
    for i in range(10):
        mgr.append("t1", "user", "A" * 30)

    assert mgr.compaction_count("t1") >= 1
    ctx = mgr.context("t1")
    # After compaction, message list must be strictly smaller than total sent.
    # New messages added after the last compact may push count above keep_messages,
    # but it must never approach the full 10 messages.
    assert len(ctx["messages"]) < 10  # type: ignore[arg-type]
    assert ctx["summary"] != ""  # summary was populated


def test_cross_session_recall(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)

    # Train both agents in thread-1 with the user's name
    adv.reply("u1", "thread-1", "Mình tên là TestUser, đang làm backend engineer.")
    base.reply("u1", "thread-1", "Mình tên là TestUser, đang làm backend engineer.")

    # Ask in a FRESH thread (cross-session recall)
    r_adv = adv.reply("u1", "thread-NEW", "Mình tên gì?")
    r_base = base.reply("u1", "thread-NEW", "Mình tên gì?")

    assert "TestUser" in r_adv["response"], (
        f"Advanced should recall name. Got: {r_adv['response']!r}"
    )
    assert "TestUser" not in r_base["response"], (
        f"Baseline should NOT recall name in new thread. Got: {r_base['response']!r}"
    )


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)

    # Send 20 turns so baseline accumulates growing context
    for i in range(20):
        msg = f"Lượt số {i}: " + "X" * 50
        adv.reply("u1", "t1", msg)
        base.reply("u1", "t1", msg)

    pt_adv = adv.prompt_token_usage("t1")
    pt_base = base.prompt_token_usage("t1")

    assert adv.compaction_count("t1") >= 1, (
        "Advanced agent should have compacted at least once."
    )
    assert pt_adv < pt_base, (
        f"Advanced prompt tokens ({pt_adv}) should be less than Baseline ({pt_base}) "
        "after compact memory activates."
    )
