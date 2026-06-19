from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


# ── UserProfileStore ──────────────────────────────────────────────────────────

def _slugify(user_id: str) -> str:
    slug = re.sub(r"[^\w\-]", "_", user_id.strip())
    return re.sub(r"_+", "_", slug).strip("_") or "unknown"


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / f"{_slugify(user_id)}.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return "# User Profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        self.write_text(user_id, current.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        return p.stat().st_size if p.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            if ":" in line and not line.startswith("#"):
                key, _, value = line.partition(":")
                k = key.strip()
                v = value.strip()
                if k and v:
                    result[k] = v
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Update existing key:value or append a new one."""
        text = self.read_text(user_id)
        pattern = re.compile(rf"^{re.escape(key)}\s*:.*$", re.MULTILINE)
        new_line = f"{key}: {value}"
        if pattern.search(text):
            text = pattern.sub(new_line, text)
        else:
            text = text.rstrip("\n") + f"\n{new_line}\n"
        self.write_text(user_id, text)


# ── extract_profile_updates ───────────────────────────────────────────────────

_CORRECTION_TRIGGERS = ("thực ra", "nhưng thực ra", "đúng hơn là", "sửa lại",
                         "chính xác hơn", "thật ra")
_NOISE_TRIGGERS = ("đùa", "câu đùa", "chỉ là đùa", "hay là chuyển sang",
                   "đùa thôi", "nói đùa")

_RE_NAME = re.compile(
    r"(?:mình\s+(?:tên|là)\s+|tên\s+(?:mình|tôi)\s+là\s+)"
    r"([A-ZÀ-Ỹa-zà-ỹ][^\.,\n?!]{1,40})",
    re.IGNORECASE,
)
_RE_LOCATION = re.compile(
    r"(?:mình\s+(?:đang\s+)?(?:ở|sống\s+tại|đang\s+làm\s+việc\s+ở|hiện\s+ở)\s+|"
    r"hiện\s+(?:đang\s+)?ở\s+|làm\s+việc\s+ở\s+)"
    r"([A-ZÀ-Ỹa-zà-ỹ][^\.,\n?!]{1,30})",
    re.IGNORECASE,
)
_RE_PROFESSION = re.compile(
    r"(?:(?:đang\s+)?làm\s+(?:nghề\s+)?|(?:^|\s)là\s+)"
    r"([A-ZÀ-Ỹa-zà-ỹ][^\.,\n?!]*"
    r"(?:engineer|developer|designer|manager|researcher|analyst|engineer|kỹ sư|lập trình)"
    r"[^\.,\n?!]{0,30})",
    re.IGNORECASE,
)
_RE_STYLE = re.compile(
    r"(?:muốn\s+bạn\s+trả\s+lời\s*|thích\s+(?:được\s+)?trả\s+lời\s*)"
    r"(.{10,100})",
    re.IGNORECASE,
)
_RE_FOOD = re.compile(
    r"món\s+ăn\s+yêu\s+thích\s+(?:là\s+|của\s+mình\s+là\s+)([^\.,\n?!]{3,40})",
    re.IGNORECASE,
)
_RE_DRINK = re.compile(
    r"(?:đồ\s+uống\s+yêu\s+thích\s+(?:là\s+|của\s+mình\s+là\s+)|"
    r"(?:yêu\s+thích\s+là\s+))([^\.,\n?!]{3,40})",
    re.IGNORECASE,
)
_RE_PET = re.compile(
    r"(?:nuôi\s+(?:một\s+con\s+|con\s+)?)([a-zA-ZÀ-Ỹà-ỹ]+(?:\s+[a-zA-ZÀ-Ỹà-ỹ]+)?)",
    re.IGNORECASE,
)


def extract_profile_updates(message: str) -> dict[str, str]:
    msg = message.strip()
    if not msg:
        return {}

    msg_lower = msg.lower()

    # Skip pure question turns
    if msg.rstrip().endswith("?") and len(msg) < 120:
        return {}
    if any(kw in msg_lower for kw in ("có phải", "bạn có biết", "bạn biết")):
        return {}

    # Detect noise / joke turns — suppress profession/location extraction
    is_noise = any(kw in msg_lower for kw in _NOISE_TRIGGERS)
    is_correction = any(kw in msg_lower for kw in _CORRECTION_TRIGGERS)

    facts: dict[str, str] = {}

    # name
    m = _RE_NAME.search(msg)
    if m:
        facts["name"] = m.group(1).strip().rstrip(".,!")

    # location — only from affirmative patterns or corrections, never from noise
    if not is_noise:
        m = _RE_LOCATION.search(msg)
        if m:
            loc = m.group(1).strip().rstrip(".,!")
            # Filter transient phrasing (vài tháng, vài ngày)
            if not re.search(r"vài\s+(?:tháng|ngày|tuần)", msg_lower[max(0, m.start()-30):m.end()+30]):
                facts["location"] = loc
            elif is_correction:
                # Explicit correction overrides even transient phrasing
                facts["location"] = loc

    # profession — skip if noise turn
    if not is_noise:
        m = _RE_PROFESSION.search(msg)
        if m:
            facts["profession"] = m.group(1).strip().rstrip(".,!")

    # style
    m = _RE_STYLE.search(msg)
    if m:
        facts["style"] = m.group(1).strip().rstrip(".,!")

    # food
    m = _RE_FOOD.search(msg)
    if m:
        facts["food"] = m.group(1).strip().rstrip(".,!")

    # drink
    m = _RE_DRINK.search(msg)
    if m:
        facts["drink"] = m.group(1).strip().rstrip(".,!")

    # pet
    m = _RE_PET.search(msg)
    if m:
        pet_val = m.group(1).strip().rstrip(".,!")
        # Only save recognisable pet words
        if re.search(r"(?:chó|mèo|corgi|lab|golden|poodle|hamster|thỏ|cat|dog)", pet_val, re.IGNORECASE):
            facts["pet"] = pet_val

    return facts


# ── summarize_messages ────────────────────────────────────────────────────────

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    recent = messages[-max_items:] if len(messages) > max_items else messages
    lines = []
    for m in recent:
        role = m.get("role", "?")
        content = m.get("content", "")
        snippet = content[:100].replace("\n", " ")
        if len(content) > 100:
            snippet += "…"
        lines.append(f"[{role}]: {snippet}")
    return "[Tóm tắt hội thoại cũ]\n" + "\n".join(lines)


# ── CompactMemoryManager ──────────────────────────────────────────────────────

@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _init_thread(self, thread_id: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._init_thread(thread_id)
        ts = self.state[thread_id]
        messages: list[dict[str, str]] = ts["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})

        total = sum(estimate_tokens(m["content"]) for m in messages)
        if total > self.threshold_tokens and len(messages) > self.keep_messages:
            old = messages[: -self.keep_messages]
            recent = messages[-self.keep_messages :]
            new_summary = summarize_messages(old)
            ts["messages"] = recent
            # Replace summary with a fresh bounded snapshot — prevents unbounded growth.
            # Long-term facts survive in User.md; compact memory only handles recent context.
            ts["summary"] = new_summary
            ts["compactions"] = int(ts["compactions"]) + 1  # type: ignore[arg-type]

    def context(self, thread_id: str) -> dict[str, object]:
        self._init_thread(thread_id)
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return int(self.state[thread_id]["compactions"])  # type: ignore[arg-type]
