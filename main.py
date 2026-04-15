import asyncio
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register


PLUGIN_NAME = "astrbot_plugin_dialog_usage_report"


@dataclass
class ModelStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    max_single_total: int = 0
    category: str = "language"  # language | image | other


@dataclass
class PendingCall:
    model: str
    provider: str
    category: str


@dataclass
class SessionWindow:
    pending_user_turns: int = 0
    rounds: int = 0
    total_calls: int = 0
    token_prompt: int = 0
    token_completion: int = 0
    token_total: int = 0
    model_stats: dict[str, ModelStats] = field(default_factory=dict)
    pending_calls: list[PendingCall] = field(default_factory=list)


@dataclass
class BotWindow:
    rounds_total: int = 0
    sessions: dict[str, SessionWindow] = field(default_factory=dict)
    observed_session_map: dict[str, str] = field(default_factory=dict)  # session_id -> umo
    admin_session_ids: set[str] = field(default_factory=set)


@register(
    PLUGIN_NAME,
    "Sakuya",
    "Bot-level token usage report for recent N rounds.",
    "1.0.1",
)
class DialogUsageReportPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._lock = asyncio.Lock()
        self._bots: dict[str, BotWindow] = {}

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_user_message(self, event: AstrMessageEvent):
        if not bool(self.config.get("enabled", True)):
            return

        text = (event.message_str or "").strip()
        if not text:
            return
        if text.startswith("/usage_report"):
            return

        umo = event.unified_msg_origin
        bot_id, msg_type, session_id = self._parse_umo(umo)
        if not self._is_monitored_session(umo, session_id):
            return

        async with self._lock:
            bot = self._ensure_bot(bot_id)
            bot.observed_session_map[session_id] = umo
            if self._is_admin_event(event):
                bot.admin_session_ids.add(session_id)

            session = self._ensure_session(bot, session_id)
            session.pending_user_turns += 1

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        if not bool(self.config.get("enabled", True)):
            return

        umo = event.unified_msg_origin
        bot_id, msg_type, session_id = self._parse_umo(umo)
        if not self._is_monitored_session(umo, session_id):
            return

        flat = self._flatten_kv(req)
        provider = self._extract_provider(flat) or await self._safe_get_provider(umo)
        model = self._extract_model(flat) or self._infer_model_from_provider(provider) or "unknown"
        category = self._detect_category(flat, model, provider)

        async with self._lock:
            bot = self._ensure_bot(bot_id)
            bot.observed_session_map[session_id] = umo
            if self._is_admin_event(event):
                bot.admin_session_ids.add(session_id)

            session = self._ensure_session(bot, session_id)
            session.total_calls += 1
            ms = session.model_stats.setdefault(model, ModelStats(category=category))
            ms.calls += 1
            ms.category = category
            session.pending_calls.append(PendingCall(model=model, provider=provider or "unknown", category=category))

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp):
        if not bool(self.config.get("enabled", True)):
            return

        umo = event.unified_msg_origin
        bot_id, msg_type, session_id = self._parse_umo(umo)
        if not self._is_monitored_session(umo, session_id):
            return

        flat = self._flatten_kv(resp)
        fallback_provider = self._extract_provider(flat) or await self._safe_get_provider(umo)
        fallback_model = self._extract_model(flat) or self._infer_model_from_provider(fallback_provider) or "unknown"
        fallback_category = self._detect_category(flat, fallback_model, fallback_provider)
        prompt_t, completion_t, total_t = self._extract_tokens(resp)
        token_sum = total_t if total_t > 0 else (prompt_t + completion_t)

        report_to_send: tuple[str, list[str], str] | None = None

        async with self._lock:
            bot = self._ensure_bot(bot_id)
            bot.observed_session_map[session_id] = umo
            if self._is_admin_event(event):
                bot.admin_session_ids.add(session_id)
            session = self._ensure_session(bot, session_id)

            if session.pending_calls:
                call = session.pending_calls.pop(0)
                model = call.model
                category = call.category
                if model == "unknown" and fallback_model != "unknown":
                    model = fallback_model
                    category = fallback_category
            else:
                model = fallback_model
                category = fallback_category
                session.total_calls += 1
                ms_missing = session.model_stats.setdefault(model, ModelStats(category=category))
                ms_missing.calls += 1

            ms = session.model_stats.setdefault(model, ModelStats(category=category))
            ms.category = category
            ms.prompt_tokens += prompt_t
            ms.completion_tokens += completion_t
            ms.total_tokens += token_sum
            if token_sum > ms.max_single_total:
                ms.max_single_total = token_sum

            session.token_prompt += prompt_t
            session.token_completion += completion_t
            session.token_total += token_sum

            if session.pending_user_turns > 0:
                session.pending_user_turns -= 1
                session.rounds += 1
                bot.rounds_total += 1

            threshold = self._normalize_rounds(self.config.get("rounds_per_report", 10))
            if bot.rounds_total >= threshold:
                report_text = self._build_bot_report(bot_id, threshold, bot)
                targets = self._resolve_report_targets(bot_id, bot)
                report_to_send = (bot_id, targets, report_text)
                self._reset_bot_window(bot_id)

        if report_to_send is not None:
            _, targets, text = report_to_send
            if not targets:
                logger.warning(f"[{PLUGIN_NAME}] no valid report targets for bot {bot_id}")
                return
            chain = MessageChain().message(text)
            for target_umo in targets:
                try:
                    await self.context.send_message(target_umo, chain)
                except Exception as exc:
                    logger.warning(f"[{PLUGIN_NAME}] failed to send report to {target_umo}: {exc}")

    @filter.command("usage_report_now")
    async def usage_report_now(self, event: AstrMessageEvent):
        """Generate report immediately for current bot."""
        umo = event.unified_msg_origin
        bot_id, msg_type, session_id = self._parse_umo(umo)

        async with self._lock:
            bot = self._bots.get(bot_id)
            if not bot:
                yield event.plain_result("当前暂无统计数据。")
                return
            threshold = self._normalize_rounds(self.config.get("rounds_per_report", 10))
            report = self._build_bot_report(bot_id, threshold, bot)
            targets = self._resolve_report_targets(bot_id, bot)
            self._reset_bot_window(bot_id)

        if not targets:
            yield event.plain_result(report)
            return

        chain = MessageChain().message(report)
        for target_umo in targets:
            try:
                await self.context.send_message(target_umo, chain)
            except Exception as exc:
                logger.warning(f"[{PLUGIN_NAME}] failed to send manual report to {target_umo}: {exc}")
        yield event.plain_result("报告已发送。")

    @filter.command("usage_report_reset")
    async def usage_report_reset(self, event: AstrMessageEvent):
        """Reset current bot window."""
        bot_id, _, _ = self._parse_umo(event.unified_msg_origin)
        async with self._lock:
            self._reset_bot_window(bot_id)
        yield event.plain_result("已重置当前 Bot 统计窗口。")

    def _build_bot_report(self, bot_id: str, rounds_n: int, bot: BotWindow) -> str:
        head = f"\U0001F4CAtoken使用报告[{bot_id}]"
        lines = [head]

        session_items = [(sid, sw) for sid, sw in bot.sessions.items() if sw.total_calls > 0]
        session_items.sort(key=lambda item: (-item[1].total_calls, item[0]))

        for session_id, sw in session_items:
            lines.append(f"\U0001F3F7\uFE0F[{session_id}]")
            lines.append(f"      期间模型调用总次数：{sw.total_calls}次")
            session_other = max(0, sw.token_total - sw.token_prompt - sw.token_completion)
            if session_other > 0:
                lines.append(
                    f"      期间Token使用总量：{self._fmt_short(sw.token_total)} [输入{self._fmt_short(sw.token_prompt)}/输出{self._fmt_short(sw.token_completion)}/其他{self._fmt_short(session_other)}]"
                )
            else:
                lines.append(
                    f"      期间Token使用总量：{self._fmt_short(sw.token_total)} [输入{self._fmt_short(sw.token_prompt)}/输出{self._fmt_short(sw.token_completion)}]"
                )
            lines.append("      调用模型名称")

            used_models: list[tuple[str, ModelStats]] = [
                (name, ms) for name, ms in sw.model_stats.items() if ms.calls > 0
            ]
            used_models.sort(key=lambda item: (-item[1].total_tokens, -item[1].calls, item[0]))

            for model_name, ms in used_models:
                avg = int(ms.total_tokens / ms.calls) if ms.calls > 0 else 0
                model_other = max(0, ms.total_tokens - ms.prompt_tokens - ms.completion_tokens)
                lines.append(f"           -{model_name}：")
                if model_other > 0:
                    lines.append(
                        f"               [输入{self._fmt_short(ms.prompt_tokens)}/输出{self._fmt_short(ms.completion_tokens)}/其他{self._fmt_short(model_other)}/合计{self._fmt_short(ms.total_tokens)}/调用次数{ms.calls}/平均调用{self._fmt_short(avg)}/最高单次{self._fmt_short(ms.max_single_total)}]"
                    )
                else:
                    lines.append(
                        f"               [输入{self._fmt_short(ms.prompt_tokens)}/输出{self._fmt_short(ms.completion_tokens)}/合计{self._fmt_short(ms.total_tokens)}/调用次数{ms.calls}/平均调用{self._fmt_short(avg)}/最高单次{self._fmt_short(ms.max_single_total)}]"
                    )

            if not used_models:
                lines.append("           -无")

        if len(lines) == 1:
            lines.append("\U0001F3F7\uFE0F[无会话数据]")
            lines.append("      期间模型调用总次数：0次")
            lines.append("      期间Token使用总量：0（输入0/输出0）")

        return "\n".join(lines)

    def _fmt_short(self, value: int) -> str:
        n = int(value or 0)
        abs_n = abs(n)
        if abs_n >= 1_000_000:
            d = (Decimal(n) / Decimal(1_000_000)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            return f"{d}m"
        if abs_n >= 10_000:
            d = (Decimal(n) / Decimal(1_000)).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
            return f"{d}k"
        return str(n)

    def _resolve_report_targets(self, bot_id: str, bot: BotWindow) -> list[str]:
        raw = self.config.get("report_target_session_ids", ["admin"])
        target_items = self._normalize_string_list(raw)
        if not target_items:
            target_items = ["admin"]

        resolved: list[str] = []

        for item in target_items:
            if item.lower() == "admin":
                for sid in sorted(bot.admin_session_ids):
                    umo = bot.observed_session_map.get(sid, "")
                    if umo:
                        resolved.append(umo)
                continue

            if item.count(":") >= 2:
                resolved.append(item)
                continue

            if item in bot.observed_session_map:
                resolved.append(bot.observed_session_map[item])

        final = []
        seen = set()
        for umo in resolved:
            _, msg_type, _ = self._parse_umo(umo)
            if msg_type == "GroupMessage":
                continue
            if umo in seen:
                continue
            seen.add(umo)
            final.append(umo)
        return final

    def _is_monitored_session(self, umo: str, session_id: str) -> bool:
        items = self._normalize_string_list(self.config.get("monitor_session_ids", []))
        if not items:
            return True
        return (session_id in items) or (umo in items)

    def _normalize_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out

    def _normalize_rounds(self, value: Any) -> int:
        try:
            n = int(value)
        except Exception:
            n = 10
        if n < 1:
            return 1
        if n > 10000:
            return 10000
        return n

    def _parse_umo(self, umo: str) -> tuple[str, str, str]:
        parts = str(umo or "").split(":", 2)
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        if len(parts) == 2:
            return parts[0], parts[1], ""
        if len(parts) == 1:
            return parts[0], "", ""
        return "", "", ""

    async def _safe_get_provider(self, umo: str) -> str:
        try:
            return await self.context.get_current_chat_provider_id(umo)
        except Exception:
            return ""

    def _ensure_bot(self, bot_id: str) -> BotWindow:
        if bot_id not in self._bots:
            self._bots[bot_id] = BotWindow()
        return self._bots[bot_id]

    def _ensure_session(self, bot: BotWindow, session_id: str) -> SessionWindow:
        if session_id not in bot.sessions:
            bot.sessions[session_id] = SessionWindow()
        return bot.sessions[session_id]

    def _reset_bot_window(self, bot_id: str):
        bot = self._bots.get(bot_id)
        if not bot:
            return
        observed = dict(bot.observed_session_map)
        admins = set(bot.admin_session_ids)
        self._bots[bot_id] = BotWindow(
            rounds_total=0,
            sessions={},
            observed_session_map=observed,
            admin_session_ids=admins,
        )

    def _is_admin_event(self, event: AstrMessageEvent) -> bool:
        try:
            method = getattr(event, "is_admin", None)
            if callable(method):
                val = method()
                if isinstance(val, bool):
                    return val
        except Exception:
            pass

        for attr_name in ("is_admin", "admin", "is_owner"):
            try:
                v = getattr(event, attr_name, None)
                if isinstance(v, bool) and v:
                    return True
            except Exception:
                pass

        try:
            message_obj = getattr(event, "message_obj", None)
            sender = getattr(message_obj, "sender", None)
            role = str(getattr(sender, "role", "")).lower()
            if role in {"admin", "owner"}:
                return True
        except Exception:
            pass

        return False

    def _extract_tokens(self, resp: Any) -> tuple[int, int, int]:
        usage_obj = getattr(resp, "usage", None)
        prompt = self._read_usage_value(usage_obj, "prompt_tokens", "input_tokens")
        completion = self._read_usage_value(usage_obj, "completion_tokens", "output_tokens")
        total = self._read_usage_value(usage_obj, "total_tokens")

        if prompt == 0 and completion == 0 and total == 0:
            usage_meta = getattr(resp, "usage_metadata", None)
            prompt = self._read_usage_value(usage_meta, "prompt_token_count", "input_token_count")
            completion = self._read_usage_value(
                usage_meta,
                "candidates_token_count",
                "completion_token_count",
                "output_token_count",
            )
            total = self._read_usage_value(usage_meta, "total_token_count")

        if prompt == 0 and completion == 0 and total == 0:
            flat = self._flatten_kv(resp)
            prompt = self._find_int_value(flat, {"prompt_tokens", "input_tokens", "prompt_token_count", "input_token_count"})
            completion = self._find_int_value(
                flat,
                {
                    "completion_tokens",
                    "output_tokens",
                    "candidates_token_count",
                    "completion_token_count",
                    "output_token_count",
                },
            )
            total = self._find_int_value(flat, {"total_tokens", "total_token_count"})

        if total == 0:
            total = prompt + completion
        return prompt, completion, total

    def _read_usage_value(self, usage: Any, *keys: str) -> int:
        for key in keys:
            if usage is None:
                continue
            if isinstance(usage, dict):
                value = usage.get(key)
            else:
                value = getattr(usage, key, None)
            try:
                if value is not None:
                    return int(float(value))
            except Exception:
                continue
        return 0

    def _flatten_kv(self, obj: Any, max_depth: int = 5) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        seen: set[int] = set()

        def walk(x: Any, depth: int):
            if depth > max_depth:
                return
            oid = id(x)
            if oid in seen:
                return
            seen.add(oid)

            if x is None:
                return
            if isinstance(x, (str, int, float, bool)):
                return

            if isinstance(x, dict):
                for k, v in x.items():
                    key = str(k).lower().strip()
                    if isinstance(v, (str, int, float, bool)):
                        out.append((key, str(v).strip()))
                    else:
                        walk(v, depth + 1)
                return

            if isinstance(x, (list, tuple, set)):
                for item in x:
                    if isinstance(item, (str, int, float, bool)):
                        out.append(("item", str(item).strip()))
                    else:
                        walk(item, depth + 1)
                return

            attrs: dict[str, Any] = {}
            try:
                attrs = vars(x)
            except Exception:
                attrs = {}

            if not attrs:
                for meth in ("model_dump", "dict"):
                    fn = getattr(x, meth, None)
                    if callable(fn):
                        try:
                            dumped = fn()
                            if isinstance(dumped, dict):
                                attrs = dumped
                                break
                        except Exception:
                            pass

            for k, v in attrs.items():
                key = str(k).lower().strip()
                if isinstance(v, (str, int, float, bool)):
                    out.append((key, str(v).strip()))
                else:
                    walk(v, depth + 1)

        walk(obj, 0)
        return out

    def _extract_provider(self, flat: list[tuple[str, str]]) -> str:
        keys = {"chat_provider_id", "provider_id", "provider", "llm_provider_id", "provider_name"}
        for key, value in flat:
            if key in keys and value:
                return value
        return ""

    def _extract_model(self, flat: list[tuple[str, str]]) -> str:
        preferred = {
            "model",
            "model_name",
            "chat_model",
            "model_id",
            "requested_model",
            "response_model",
            "used_model",
            "model_version",
            "model_name_or_path",
        }
        for key, value in flat:
            if key in preferred and self._looks_like_model(value):
                return value

        for key, value in flat:
            if "model" not in key:
                continue
            if any(x in key for x in ("provider", "embedding", "mode")):
                continue
            if self._looks_like_model(value):
                return value
        return ""

    def _looks_like_model(self, value: str) -> bool:
        v = str(value or "").strip()
        if not v:
            return False
        low = v.lower()
        if low in {"true", "false", "none", "null"}:
            return False
        if low.isdigit():
            return False
        if any(x in low for x in ("provider", "http://", "https://")):
            return False
        return True

    def _infer_model_from_provider(self, provider: str) -> str:
        p = str(provider or "").strip()
        if not p:
            return ""
        if "/" in p:
            candidate = p.split("/")[-1].strip()
            if self._looks_like_model(candidate):
                return candidate
        return ""

    def _detect_category(self, flat: list[tuple[str, str]], model: str, provider: str) -> str:
        hint = self._find_value_by_key(flat, {"modalities", "modality", "mode"}).lower()
        if "image" in hint or "vision" in hint:
            return "image"
        if "video" in hint:
            return "other"
        if any(x in hint for x in ("audio", "speech", "voice", "stt", "tts", "asr")):
            return "other"

        text = f"{model} {provider}".lower()
        if any(k in text for k in ("image", "vision", "dall", "sd", "midjourney", "flux")):
            return "image"
        if any(k in text for k in ("video", "audio", "speech", "voice", "stt", "tts", "asr", "whisper")):
            return "other"
        return "language"

    def _find_value_by_key(self, flat: list[tuple[str, str]], keys: set[str]) -> str:
        for key, value in flat:
            if key in keys and value:
                return value
        return ""

    def _find_int_value(self, flat: list[tuple[str, str]], keys: set[str]) -> int:
        for key, value in flat:
            if key not in keys:
                continue
            try:
                return int(float(value))
            except Exception:
                continue
        return 0
