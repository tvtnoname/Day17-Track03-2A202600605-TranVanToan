from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only. No User.md, no cross-thread recall."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    # ── public API ────────────────────────────────────────────────────────────

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        return self._reply_live(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    # ── offline path ──────────────────────────────────────────────────────────

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        state = self.sessions[thread_id]

        state.messages.append({"role": "user", "content": message})

        reply = self._lookup_reply(state.messages, message)
        state.messages.append({"role": "assistant", "content": reply})

        agent_toks = estimate_tokens(message) + estimate_tokens(reply)
        prompt_toks = sum(estimate_tokens(m["content"]) for m in state.messages)

        state.token_usage += agent_toks
        state.prompt_tokens_processed += prompt_toks

        return {
            "response": reply,
            "agent_tokens": agent_toks,
            "prompt_tokens": prompt_toks,
        }

    @staticmethod
    def _lookup_reply(messages: list[dict[str, str]], message: str) -> str:
        msg_lower = message.lower()
        # Only search within the current thread
        facts: dict[str, str] = {}
        for m in messages:
            if m["role"] != "user":
                continue
            txt = m["content"]
            _extract_inline(txt, facts)

        # Answer multi-fact questions by collecting everything asked
        answers: list[str] = []

        if any(kw in msg_lower for kw in ("tên", "là ai", "mình là")):
            if "name" in facts:
                answers.append(f"Tên bạn là {facts['name']}")

        if any(kw in msg_lower for kw in ("nghề", "làm gì", "công việc", "nghề nghiệp")):
            if "profession" in facts:
                answers.append(f"Nghề nghiệp: {facts['profession']}")

        if any(kw in msg_lower for kw in ("ở đâu", "nơi ở", "đang ở", "sống")):
            if "location" in facts:
                answers.append(f"Nơi ở: {facts['location']}")

        if any(kw in msg_lower for kw in ("uống", "đồ uống")):
            if "drink" in facts:
                answers.append(f"Đồ uống yêu thích: {facts['drink']}")

        if any(kw in msg_lower for kw in ("ăn", "món ăn")):
            if "food" in facts:
                answers.append(f"Món ăn yêu thích: {facts['food']}")

        if any(kw in msg_lower for kw in ("style", "trả lời", "phong cách")):
            if "style" in facts:
                answers.append(f"Style trả lời: {facts['style']}")

        if any(kw in msg_lower for kw in ("nuôi", "thú cưng", "pet")):
            if "pet" in facts:
                answers.append(f"Thú cưng: {facts['pet']}")

        if answers:
            return ". ".join(answers) + "."
        return "Xin lỗi, mình không có thông tin đó trong cuộc hội thoại này."

    # ── live path ─────────────────────────────────────────────────────────────

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        try:
            from langchain_core.messages import HumanMessage
            config = {"configurable": {"thread_id": thread_id}}
            result = self.langchain_agent.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config,
            )
            msgs = result.get("messages", [])
            reply = msgs[-1].content if msgs else ""
            agent_toks, prompt_toks = _extract_token_usage(msgs)
            if thread_id not in self.sessions:
                self.sessions[thread_id] = SessionState()
            self.sessions[thread_id].token_usage += agent_toks
            self.sessions[thread_id].prompt_tokens_processed += prompt_toks
            return {"response": reply, "agent_tokens": agent_toks, "prompt_tokens": prompt_toks}
        except Exception as e:
            print(f"[Baseline live error] {e!r} — falling back to offline")
            return self._reply_offline(thread_id, message)

    def _maybe_build_langchain_agent(self) -> None:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            from model_provider import build_chat_model

            llm = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(llm, tools=[], checkpointer=checkpointer)
        except Exception:
            self.langchain_agent = None


# ── helpers ───────────────────────────────────────────────────────────────────

import re as _re


def _extract_token_usage(msgs: list) -> tuple[int, int]:
    """Return (agent_tokens, prompt_tokens) from LangChain message list.

    Reads `usage_metadata` (LangChain standard) or falls back to heuristic estimate.
    """
    agent_toks = 0
    prompt_toks = 0
    for m in msgs:
        meta = getattr(m, "usage_metadata", None)
        if meta:
            agent_toks += getattr(meta, "output_tokens", 0)
            prompt_toks += getattr(meta, "input_tokens", 0)
    if agent_toks == 0 and msgs:
        # fallback: heuristic from content length
        agent_toks = estimate_tokens(msgs[-1].content if msgs else "")
        prompt_toks = sum(estimate_tokens(getattr(m, "content", "")) for m in msgs)
    return agent_toks, prompt_toks


def _extract_inline(text: str, facts: dict[str, str]) -> None:
    """Best-effort inline fact extraction for baseline (within-thread only)."""
    t = text.lower()

    m = _re.search(r"(?:mình\s+(?:tên|là)\s+|tên\s+(?:mình|tôi)\s+là\s+)"
                   r"([A-ZÀ-Ỹa-zà-ỹ][^\.,\n?!]{1,40})", text, _re.IGNORECASE)
    if m:
        facts["name"] = m.group(1).strip().rstrip(".,!")

    m = _re.search(r"(?:mình\s+(?:đang\s+)?(?:ở|sống\s+tại|đang\s+làm\s+việc\s+ở|hiện\s+ở)\s+|"
                   r"hiện\s+(?:đang\s+)?ở\s+|làm\s+việc\s+ở\s+)"
                   r"([A-ZÀ-Ỹa-zà-ỹ][^\.,\n?!]{1,30})", text, _re.IGNORECASE)
    if m:
        facts["location"] = m.group(1).strip().rstrip(".,!")

    m = _re.search(r"(?:(?:đang\s+)?làm\s+(?:nghề\s+)?|(?:^|\s)là\s+)"
                   r"([A-ZÀ-Ỹa-zà-ỹ][^\.,\n?!]*"
                   r"(?:engineer|developer|designer|manager|researcher|analyst|kỹ sư)"
                   r"[^\.,\n?!]{0,30})", text, _re.IGNORECASE)
    if m:
        facts["profession"] = m.group(1).strip().rstrip(".,!")

    m = _re.search(r"(?:đồ\s+uống\s+yêu\s+thích\s+(?:là\s+|của\s+mình\s+là\s+)|"
                   r"(?:yêu\s+thích\s+là\s+))([^\.,\n?!]{3,40})", text, _re.IGNORECASE)
    if m:
        facts["drink"] = m.group(1).strip().rstrip(".,!")

    m = _re.search(r"món\s+ăn\s+yêu\s+thích\s+(?:là\s+|của\s+mình\s+là\s+)"
                   r"([^\.,\n?!]{3,40})", text, _re.IGNORECASE)
    if m:
        facts["food"] = m.group(1).strip().rstrip(".,!")

    m = _re.search(r"(?:muốn\s+bạn\s+trả\s+lời\s*)(.{10,100})", text, _re.IGNORECASE)
    if m:
        facts["style"] = m.group(1).strip().rstrip(".,!")

    m = _re.search(r"(?:nuôi\s+(?:một\s+con\s+|con\s+)?)"
                   r"([a-zA-ZÀ-Ỹà-ỹ]+(?:\s+[a-zA-ZÀ-Ỹà-ỹ]+)?)", text, _re.IGNORECASE)
    if m:
        pet_val = m.group(1).strip()
        if _re.search(r"(?:chó|mèo|corgi|lab|golden|poodle|hamster|thỏ|cat|dog)",
                      pet_val, _re.IGNORECASE):
            facts["pet"] = pet_val
