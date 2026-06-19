from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: 3-layer memory — short-term, persistent User.md, compact."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    # ── public API ────────────────────────────────────────────────────────────

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        return self._reply_live(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # ── offline path ──────────────────────────────────────────────────────────

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1. Extract and persist facts
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 2. Append user turn to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context load BEFORE generating reply
        prompt_toks = self._estimate_prompt_context_tokens(user_id, thread_id)

        # 4. Generate deterministic reply using persisted memory
        reply_text = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant reply
        self.compact_memory.append(thread_id, "assistant", reply_text)

        # 6. Update counters
        agent_toks = estimate_tokens(message) + estimate_tokens(reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_toks
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_toks
        )

        return {
            "response": reply_text,
            "agent_tokens": agent_toks,
            "prompt_tokens": prompt_toks,
        }

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:  # noqa: ARG002
        facts = self.profile_store.facts(user_id)
        msg_lower = message.lower()

        answers: list[str] = []

        if any(kw in msg_lower for kw in ("tên", "là ai", "mình là", "tên mình")):
            if "name" in facts:
                answers.append(f"Tên bạn là {facts['name']}")

        if any(kw in msg_lower for kw in ("nghề", "làm gì", "công việc", "nghề nghiệp",
                                           "nghề hiện tại", "mlops", "engineer")):
            if "profession" in facts:
                answers.append(f"Nghề nghiệp hiện tại: {facts['profession']}")

        if any(kw in msg_lower for kw in ("ở đâu", "nơi ở", "đang ở", "sống", "địa điểm",
                                           "nơi ở hiện tại")):
            if "location" in facts:
                answers.append(f"Nơi ở hiện tại: {facts['location']}")

        if any(kw in msg_lower for kw in ("uống", "đồ uống", "thức uống")):
            if "drink" in facts:
                answers.append(f"Đồ uống yêu thích: {facts['drink']}")

        if any(kw in msg_lower for kw in ("ăn", "món ăn", "thức ăn")):
            if "food" in facts:
                answers.append(f"Món ăn yêu thích: {facts['food']}")

        if any(kw in msg_lower for kw in ("style", "trả lời", "phong cách", "kiểu trả lời")):
            if "style" in facts:
                answers.append(f"Style trả lời bạn thích: {facts['style']}")

        if any(kw in msg_lower for kw in ("nuôi", "thú cưng", "pet", "con gì")):
            if "pet" in facts:
                answers.append(f"Thú cưng: {facts['pet']}")

        # Check if question asks about "all info" (tóm tắt, nhắc lại, etc.)
        if any(kw in msg_lower for kw in ("nhắc lại", "tóm tắt", "thông tin")):
            for key, label in [("name", "Tên"), ("profession", "Nghề"),
                                ("location", "Nơi ở"), ("drink", "Đồ uống"),
                                ("food", "Món ăn"), ("style", "Style trả lời"),
                                ("pet", "Thú cưng")]:
                if key in facts and not any(label.lower() in a.lower() for a in answers):
                    answers.append(f"{label}: {facts[key]}")

        if answers:
            return ". ".join(answers) + "."

        # Fallback: let user know we have their profile but can't answer this specifically
        if len(facts) > 0:
            return f"Mình có thông tin về bạn trong hồ sơ nhưng câu hỏi này mình chưa có dữ liệu cụ thể."
        return "Mình chưa có thông tin đó trong hồ sơ của bạn."

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_tokens = estimate_tokens(self.profile_store.read_text(user_id))
        ctx = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(str(ctx.get("summary", "")))
        msg_tokens = sum(
            estimate_tokens(m["content"])
            for m in ctx.get("messages", [])  # type: ignore[union-attr]
        )
        return profile_tokens + summary_tokens + msg_tokens

    # ── live path ─────────────────────────────────────────────────────────────

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            profile_text = self.profile_store.read_text(user_id)
            ctx = self.compact_memory.context(thread_id)
            summary = str(ctx.get("summary", ""))

            system_parts = [
                f"Bạn đang trò chuyện với người dùng có ID: {user_id}",
                f"Hồ sơ người dùng:\n{profile_text}",
            ]
            if summary:
                system_parts.append(f"Tóm tắt hội thoại cũ:\n{summary}")
            system_content = "\n\n".join(system_parts)

            config = {"configurable": {"thread_id": thread_id}}
            result = self.langchain_agent.invoke(
                {"messages": [SystemMessage(content=system_content),
                              HumanMessage(content=message)]},
                config=config,
            )
            msgs = result.get("messages", [])
            reply = msgs[-1].content if msgs else ""

            # Update User.md with extracted facts
            updates = extract_profile_updates(message)
            for key, value in updates.items():
                self.profile_store.upsert_fact(user_id, key, value)

            self.compact_memory.append(thread_id, "user", message)
            self.compact_memory.append(thread_id, "assistant", reply)

            from agent_baseline import _extract_token_usage
            agent_toks, prompt_toks_api = _extract_token_usage(msgs)
            # Use API prompt tokens if available; otherwise fall back to context estimate
            prompt_toks = prompt_toks_api or self._estimate_prompt_context_tokens(user_id, thread_id)
            self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_toks
            self.thread_prompt_tokens[thread_id] = (
                self.thread_prompt_tokens.get(thread_id, 0) + prompt_toks
            )
            return {"response": reply, "agent_tokens": agent_toks, "prompt_tokens": prompt_toks}
        except Exception as e:
            print(f"[Advanced live error] {e!r} — falling back to offline")
            return self._reply_offline(user_id, thread_id, message)

    def _maybe_build_langchain_agent(self) -> None:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            from model_provider import build_chat_model

            llm = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            # No write tool — profile updates are handled by Python code in _reply_live
            # (extract_profile_updates → upsert_fact). Giving the LLM a write tool causes
            # it to invent wrong user_ids and corrupt User.md.
            self.langchain_agent = create_react_agent(llm, tools=[], checkpointer=checkpointer)
        except Exception:
            self.langchain_agent = None
