"""
astrbot_plugin_dialog_usage_report
对话 Token 使用报告插件

本插件95%由Codex和Claude互相监督。

功能概述：
  - 监听指定会话（或全部会话）的 LLM 请求与响应，按 Bot 实例分别统计：
      · 语言模型的 Token 使用量（输入/输出/其他/合计）
  - 支持两种自动触发方式：
      · 按对话轮次触发（可配置）
      · 按时间间隔触发（可配置）
  - 报告发送目标可配置为 AstrBot 全局管理员（admin）或指定会话 ID
  - 支持手动命令查询最近 N 轮或最近 X 分钟的统计（可能存在一点点小bug，还在测试中）

"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register


# 插件唯一标识符，与插件目录名一致
PLUGIN_NAME = "astrbot_plugin_dialog_usage_report"


# ──────────────────────────────────────────────
# 数据结构定义
# ──────────────────────────────────────────────

@dataclass
class LanguageStats:
    """单个 Provider 在某会话内的聚合 Token 统计（用于构建报告）。"""
    calls: int = 0                  # 调用次数
    prompt_tokens: int = 0          # 输入 Token 总量
    completion_tokens: int = 0      # 输出 Token 总量
    other_tokens: int = 0           # 其他 Token（如缓存命中等）
    total_tokens: int = 0           # 总 Token（含其他）
    max_single_total: int = 0       # 单次最高 Token 消耗
    image_reads: int = 0            # 多模态图片读取次数
    audio_reads: int = 0            # 多模态音频读取次数


@dataclass
class LanguageCallRecord:
    """一次 LLM 调用的原始记录（语言模型）。"""
    ts: datetime                    # 记录时间（UTC）
    session_id: str                 # 会话 ID
    chat_kind: str                  # 会话类型："group" 或 "private"
    provider_id: str                # Provider 标识
    prompt_tokens: int              # 本次输入 Token
    completion_tokens: int          # 本次输出 Token
    other_tokens: int               # 本次其他 Token
    total_tokens: int               # 本次总 Token
    image_reads: int                # 本次读取图片数
    audio_reads: int                # 本次读取音频数
    round_id: int                   # 对话轮次 ID（0 表示未关联到完整轮次）


@dataclass
class SpeechGenRecord:
    """一次语音生成（TTS）的原始记录。"""
    ts: datetime                    # 记录时间（UTC）
    session_id: str
    chat_kind: str
    provider_id: str
    chars: int                      # 输入文字字符数
    duration_seconds: float         # 生成音频时长（秒）
    tokens: int                     # 消耗 Token（部分 TTS 服务提供）
    round_id: int


@dataclass
class ImageGenRecord:
    """一次图片生成的原始记录。"""
    ts: datetime                    # 记录时间（UTC）
    session_id: str
    chat_kind: str
    provider_id: str
    image_count: int                # 本次生成图片张数
    round_id: int


@dataclass
class PendingCall:
    """在 on_llm_request 与 on_llm_response 之间传递的中间状态。
    
    由于 AstrBot 的请求/响应钩子是分开触发的，需要在请求阶段记录
    Provider 信息和多模态读取数量，等响应阶段再合并写入记录。
    """
    provider_id: str
    image_reads: int
    audio_reads: int


@dataclass
class SessionState:
    """单个会话的中间状态，用于轮次匹配。"""
    # 尚未被 LLM 响应消耗的用户发言计数（用于判断是否构成完整轮次）
    pending_user_turns: int = 0
    # 已发出但尚未收到响应的 LLM 请求列表（保留 Provider 信息）
    pending_calls: list[PendingCall] = field(default_factory=list)
    # 本会话最后一次完整轮次的 ID（供 after_message_sent 关联媒体记录）
    last_round_id: int = 0


@dataclass
class BotState:
    """单个 Bot 实例（bot_id）的全量运行时状态。
    
    AstrBot 可能同时运行多个平台适配器（Bot），各 Bot 的数据相互独立。
    """
    # 会话状态表：session_id → SessionState
    sessions: dict[str, SessionState] = field(default_factory=dict)

    # 原始记录列表（最多保留最近 3 天）
    language_records: list[LanguageCallRecord] = field(default_factory=list)
    speech_records: list[SpeechGenRecord] = field(default_factory=list)
    image_records: list[ImageGenRecord] = field(default_factory=list)

    # 全局轮次计数器（单调递增，跨会话）
    round_counter: int = 0
    # 轮次事件列表：(round_id, 发生时间)，用于按轮次查询
    round_events: list[tuple[int, datetime]] = field(default_factory=list)

    # 已观察到的所有会话映射：session_id → unified_msg_origin（完整 UMO）
    # 用于将 session_id 解析为可直接发送消息的 UMO 目标
    # 注意：此映射在 _is_monitored 检查之前写入，以确保报告目标（如管理员）
    # 即使不在监控范围内也能被正确寻址
    observed_session_map: dict[str, str] = field(default_factory=dict)

    # 本 Bot 下已确认为 AstrBot 全局管理员的 session_id 集合
    # 判断依据：发送者 user_id 在 AstrBot 全局配置的 admins_id 列表中
    admin_session_ids: set[str] = field(default_factory=set)

    # ── 轮次触发模式的窗口状态 ──
    # 当前轮次触发窗口的起始时间（群聊/私聊分开计算）
    auto_window_start_group: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    auto_window_start_private: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # 当前窗口内累计的完整对话轮次数
    auto_rounds_group: int = 0
    auto_rounds_private: int = 0

    # 是否已触发但尚未发送的轮次报告标志（等待 after_message_sent 再发送，
    # 确保当轮媒体记录也已写入）
    pending_round_trigger_group: bool = False
    pending_round_trigger_private: bool = False

    # ── 时间触发模式的窗口状态 ──
    time_window_start_group: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    time_window_start_private: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # 当前时间窗口内累计的完整对话轮次数（至少有 1 轮才发送，避免空报告）
    time_rounds_group: int = 0
    time_rounds_private: int = 0


# ──────────────────────────────────────────────
# 插件主类
# ──────────────────────────────────────────────

@register(
    PLUGIN_NAME,
    "Sakuya_mzk",
    "按 Bot 统计模型调用与 Token 使用情况，支持轮次/时间触发自动报告。",
    "1.1.0",
)
class DialogUsageReportPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # 异步锁：保护对 _bots 字典及其内部状态的并发访问
        self._lock = asyncio.Lock()
        # 各 Bot 的运行时状态：bot_id → BotState
        self._bots: dict[str, BotState] = {}
        # 定时报告的后台任务句柄
        self._timer_task: asyncio.Task | None = None
        self._timer_running: bool = False

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """AstrBot 启动完成后立即启动定时报告调度器。"""
        await self._start_timer_if_needed()

    async def terminate(self):
        """插件卸载/重载时优雅停止定时器任务。"""
        self._timer_running = False
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
        self._timer_task = None

    # ──────────────────────────────────────────────
    # 事件监听器
    # ──────────────────────────────────────────────

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_user_message(self, event: AstrMessageEvent):
        """监听所有用户消息，完成两件事：
        1. 无论是否在监控范围内，都记录 session_id → UMO 映射和管理员身份，
           以便报告目标解析时能找到管理员的 UMO。
        2. 仅对监控范围内的会话累加待处理用户发言计数，用于后续轮次匹配。
        """
        await self._start_timer_if_needed()
        if not bool(self.config.get("enabled", True)):
            return
        text = (event.message_str or "").strip()
        # 忽略空消息
        if not text:
            return
        # 忽略 token报告 命令本身（避免命令触发被计入轮次）
        if text.startswith("/token报告") or text.startswith("token报告"):
            return

        umo = event.unified_msg_origin
        bot_id, msg_type, session_id = self._parse_umo(umo)
        chat_kind = self._chat_kind(msg_type)
        if not self._is_chat_kind_enabled(chat_kind):
            return

        # 第一步：无论是否受监控，先更新全局会话映射和管理员标记
        async with self._lock:
            bot = self._ensure_bot(bot_id)
            bot.observed_session_map[session_id] = umo
            if self._is_admin_event(event):
                bot.admin_session_ids.add(session_id)

        # 第二步：仅对受监控会话计入待处理用户发言
        if not self._is_monitored(umo, session_id):
            return

        async with self._lock:
            bot = self._ensure_bot(bot_id)
            self._ensure_session(bot, session_id).pending_user_turns += 1

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """在 LLM 请求发出前，记录 Provider 信息和多模态资产数量。
        
        请求和响应是分开触发的钩子，此处将信息暂存到 pending_calls 队列，
        等响应返回后在 on_llm_response 中取出合并。
        """
        await self._start_timer_if_needed()
        if not bool(self.config.get("enabled", True)):
            return
        umo = event.unified_msg_origin
        bot_id, msg_type, session_id = self._parse_umo(umo)
        chat_kind = self._chat_kind(msg_type)
        if not self._is_chat_kind_enabled(chat_kind):
            return

        # 同样先更新全局会话映射和管理员标记（兼容仅有 LLM 交互无普通消息的场景）
        async with self._lock:
            bot = self._ensure_bot(bot_id)
            bot.observed_session_map[session_id] = umo
            if self._is_admin_event(event):
                bot.admin_session_ids.add(session_id)

        if not self._is_monitored(umo, session_id):
            return

        flat = self._flatten_kv(req)
        provider = self._extract_provider(flat) or await self._safe_get_provider(umo)
        provider_id = self._normalize_provider_id(provider, fallback="unknown_llm")

        # 保守统计多模态读取：只识别明确的图片/音频输入字段，
        # 避免误把模型名称等字符串当作图片路径计数。
        image_reads = self._count_explicit_assets(req, "image")
        audio_reads = self._count_explicit_assets(req, "audio")

        async with self._lock:
            bot = self._ensure_bot(bot_id)
            self._ensure_session(bot, session_id).pending_calls.append(
                PendingCall(
                    provider_id=provider_id,
                    image_reads=image_reads,
                    audio_reads=audio_reads,
                )
            )

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp):
        """LLM 响应返回后，提取 Token 用量并写入记录，同时判断是否触发自动报告。"""
        await self._start_timer_if_needed()
        if not bool(self.config.get("enabled", True)):
            return

        umo = event.unified_msg_origin
        bot_id, msg_type, session_id = self._parse_umo(umo)
        chat_kind = self._chat_kind(msg_type)

        # 更新全局会话映射（与 on_llm_request 保持一致）
        async with self._lock:
            bot = self._ensure_bot(bot_id)
            bot.observed_session_map[session_id] = umo
            if self._is_admin_event(event):
                bot.admin_session_ids.add(session_id)

        if not self._is_monitored(umo, session_id):
            return
        if not self._is_chat_kind_enabled(chat_kind):
            return

        flat = self._flatten_kv(resp)
        provider = self._extract_provider(flat) or await self._safe_get_provider(umo)
        fallback_provider_id = self._normalize_provider_id(
            provider, fallback="unknown_llm"
        )
        prompt_tokens, completion_tokens, total_tokens = self._extract_tokens(resp)
        total = (
            total_tokens if total_tokens > 0 else (prompt_tokens + completion_tokens)
        )
        # 计算"其他"Token（如缓存命中 Token 等，总量超出输入+输出的部分）
        other = max(0, total - prompt_tokens - completion_tokens)

        async with self._lock:
            bot = self._ensure_bot(bot_id)
            session = self._ensure_session(bot, session_id)

            # 从请求阶段暂存的 pending_calls 中取出对应的 Provider 信息
            provider_id = fallback_provider_id
            image_reads = 0
            audio_reads = 0
            if session.pending_calls:
                call = session.pending_calls.pop(0)
                provider_id = call.provider_id or fallback_provider_id
                image_reads = call.image_reads
                audio_reads = call.audio_reads

            # 判断本次响应是否构成完整对话轮次（有对应的用户发言则消耗一个）
            round_id = 0
            if session.pending_user_turns > 0:
                session.pending_user_turns -= 1
                bot.round_counter += 1
                round_id = bot.round_counter
                session.last_round_id = round_id
                bot.round_events.append((round_id, datetime.now(timezone.utc)))
                # 分别累计群聊和私聊的轮次数（轮次触发和时间触发各用一套计数）
                if chat_kind == "group":
                    bot.auto_rounds_group += 1
                    bot.time_rounds_group += 1
                else:
                    bot.auto_rounds_private += 1
                    bot.time_rounds_private += 1

            # 写入本次调用记录
            bot.language_records.append(
                LanguageCallRecord(
                    ts=datetime.now(timezone.utc),
                    session_id=session_id,
                    chat_kind=chat_kind,
                    provider_id=provider_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    other_tokens=other,
                    total_tokens=total,
                    image_reads=image_reads,
                    audio_reads=audio_reads,
                    round_id=round_id,
                )
            )
            # 清理超过 3 天的历史数据，防止内存无限增长
            self._prune_history(bot)

            # 检查是否达到轮次触发阈值，若是则标记待发送
            # （实际发送推迟到 after_message_sent，确保当轮媒体记录也已写入）
            if self._should_trigger_round(bot, chat_kind):
                if chat_kind == "group":
                    bot.pending_round_trigger_group = True
                else:
                    bot.pending_round_trigger_private = True

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent):
        """消息发送完毕后：
        1. 捕获本轮回复中包含的语音/图片生成记录。
        2. 若之前已标记轮次触发，在此时发送自动报告（此时当轮所有记录均已写入）。
        """
        await self._start_timer_if_needed()
        if not bool(self.config.get("enabled", True)):
            return

        umo = event.unified_msg_origin
        bot_id, msg_type, session_id = self._parse_umo(umo)
        chat_kind = self._chat_kind(msg_type)
        if not self._is_chat_kind_enabled(chat_kind):
            return
        if not self._is_monitored(umo, session_id):
            return

        # 从本次发送的消息链中提取语音和图片生成信息
        speech_items, image_items = self._capture_generated_media(event)

        async with self._lock:
            bot = self._bots.get(bot_id)
            if not bot:
                return
            session = self._ensure_session(bot, session_id)
            round_id = session.last_round_id

            # 写入语音生成记录
            for item in speech_items:
                bot.speech_records.append(
                    SpeechGenRecord(
                        ts=datetime.now(timezone.utc),
                        session_id=session_id,
                        chat_kind=chat_kind,
                        provider_id=item["provider_id"],
                        chars=item["chars"],
                        duration_seconds=item["duration_seconds"],
                        tokens=item["tokens"],
                        round_id=round_id,
                    )
                )

            # 写入图片生成记录
            for item in image_items:
                bot.image_records.append(
                    ImageGenRecord(
                        ts=datetime.now(timezone.utc),
                        session_id=session_id,
                        chat_kind=chat_kind,
                        provider_id=item["provider_id"],
                        image_count=item["image_count"],
                        round_id=round_id,
                    )
                )

            if speech_items or image_items:
                self._prune_history(bot)

            # 读取当前聊天类型的轮次触发标志
            should_send = (
                bot.pending_round_trigger_group
                if chat_kind == "group"
                else bot.pending_round_trigger_private
            )

        if not should_send:
            return

        # 发送自动报告（轮次模式）
        sent = await self._send_auto_report(bot_id, chat_kind, trigger_mode="round")
        if sent:
            async with self._lock:
                bot = self._bots.get(bot_id)
                if not bot:
                    return
                # 清除触发标志，防止重复发送
                if chat_kind == "group":
                    bot.pending_round_trigger_group = False
                else:
                    bot.pending_round_trigger_private = False

    # ──────────────────────────────────────────────
    # 定时报告调度器
    # ──────────────────────────────────────────────

    async def _start_timer_if_needed(self):
        """幂等启动定时报告后台任务。若任务已在运行则直接返回。"""
        if self._timer_running and self._timer_task and not self._timer_task.done():
            return
        self._timer_running = True
        self._timer_task = asyncio.create_task(self._timer_loop())
        logger.info(f"[{PLUGIN_NAME}] 定时报告调度器已启动")

    def _timer_tick_seconds(self) -> int:
        """根据当前配置动态计算定时器轮询间隔（秒）。
        
        取群聊/私聊时间触发窗口的最小值的 1/6 作为轮询粒度，
        并限定在 [15, 60] 秒之间，兼顾响应速度与资源开销。
        """
        mins: list[int] = []
        g = self._parse_optional_positive(
            self.config.get("group_time_trigger_minutes", ""), 1440
        )
        p = self._parse_optional_positive(
            self.config.get("private_time_trigger_minutes", ""), 1440
        )
        if self._is_chat_kind_enabled("group") and g is not None:
            mins.append(g)
        if self._is_chat_kind_enabled("private") and p is not None:
            mins.append(p)
        if not mins:
            return 60
        sec = int(min(mins) * 60 / 6)
        return max(15, min(60, sec))

    async def _timer_loop(self):
        """定时报告主循环：按配置的时间窗口检查并触发自动报告。"""
        while self._timer_running:
            try:
                await asyncio.sleep(self._timer_tick_seconds())
                if not bool(self.config.get("enabled", True)):
                    continue

                # 收集需要发送报告的 (bot_id, chat_kind) 组合
                to_send: list[tuple[str, str]] = []
                now = datetime.now(timezone.utc)
                async with self._lock:
                    for bot_id, bot in self._bots.items():
                        g_minutes = self._parse_optional_positive(
                            self.config.get("group_time_trigger_minutes", ""), 1440
                        )
                        p_minutes = self._parse_optional_positive(
                            self.config.get("private_time_trigger_minutes", ""), 1440
                        )

                        # 群聊时间触发检查
                        if (
                            self._is_chat_kind_enabled("group")
                            and g_minutes is not None
                            and (now - bot.time_window_start_group).total_seconds()
                            >= g_minutes * 60
                        ):
                            if bot.time_rounds_group >= 1:
                                # 有数据才发送
                                to_send.append((bot_id, "group"))
                            else:
                                # 无对话记录，静默重置窗口
                                bot.time_window_start_group = now

                        # 私聊时间触发检查
                        if (
                            self._is_chat_kind_enabled("private")
                            and p_minutes is not None
                            and (now - bot.time_window_start_private).total_seconds()
                            >= p_minutes * 60
                        ):
                            if bot.time_rounds_private >= 1:
                                to_send.append((bot_id, "private"))
                            else:
                                bot.time_window_start_private = now

                for bot_id, chat_kind in to_send:
                    await self._send_auto_report(bot_id, chat_kind, trigger_mode="time")
            except asyncio.CancelledError:
                logger.info(f"[{PLUGIN_NAME}] 定时报告调度器已取消")
                break
            except Exception as exc:
                logger.warning(f"[{PLUGIN_NAME}] 定时器循环异常: {exc}")
                await asyncio.sleep(60)

    # ──────────────────────────────────────────────
    # 自动报告发送
    # ──────────────────────────────────────────────

    async def _send_auto_report(
        self, bot_id: str, chat_kind: str, trigger_mode: str = "round"
    ) -> bool:
        """构建并发送自动报告。
        
        Args:
            bot_id: Bot 实例标识
            chat_kind: 会话类型（"group" 或 "private"）
            trigger_mode: 触发方式（"round" 或 "time"）
        
        Returns:
            True 表示至少成功发送给一个目标，False 表示未发送。
        """
        async with self._lock:
            bot = self._bots.get(bot_id)
            if not bot:
                return False

            # 根据触发模式确定统计窗口起始时间
            if trigger_mode == "time":
                start = (
                    bot.time_window_start_group
                    if chat_kind == "group"
                    else bot.time_window_start_private
                )
            else:
                start = (
                    bot.auto_window_start_group
                    if chat_kind == "group"
                    else bot.auto_window_start_private
                )

            # 筛选窗口内的记录
            language_records = [
                r
                for r in bot.language_records
                if r.chat_kind == chat_kind and r.ts >= start
            ]
            speech_records = [
                r
                for r in bot.speech_records
                if r.chat_kind == chat_kind and r.ts >= start
            ]
            image_records = [
                r
                for r in bot.image_records
                if r.chat_kind == chat_kind and r.ts >= start
            ]

            # 无记录则不发送
            if not language_records and not speech_records and not image_records:
                return False

            report = self._build_report(
                bot_id=bot_id,
                language_records=language_records,
                speech_records=speech_records,
                image_records=image_records,
                include_rounds=True,
            )
            targets = self._resolve_report_targets(bot, bot_id)

        if not targets:
            logger.warning(
                f"[{PLUGIN_NAME}] 自动报告无有效发送目标，bot={bot_id}。"
                "请检查全局管理员配置（Settings -> Other Settings -> Admin ID List）"
                "或 report_target_session_ids 插件配置。"
            )
            return False

        chain = MessageChain().message(report)
        sent_any = False
        for target in targets:
            try:
                await self.context.send_message(target, chain)
                sent_any = True
            except Exception as exc:
                logger.warning(f"[{PLUGIN_NAME}] 发送报告失败，目标={target}: {exc}")
        if not sent_any:
            return False

        # 重置对应窗口的统计计数
        async with self._lock:
            bot = self._bots.get(bot_id)
            if not bot:
                return True
            now = datetime.now(timezone.utc)
            if trigger_mode == "time":
                if chat_kind == "group":
                    bot.time_window_start_group = now
                    bot.time_rounds_group = 0
                else:
                    bot.time_window_start_private = now
                    bot.time_rounds_private = 0
            else:
                if chat_kind == "group":
                    bot.auto_window_start_group = now
                    bot.auto_rounds_group = 0
                else:
                    bot.auto_window_start_private = now
                    bot.auto_rounds_private = 0
        return True

    # ──────────────────────────────────────────────
    # 手动查询命令
    # ──────────────────────────────────────────────

    @filter.command("token报告")
    async def token_report(
        self, event: AstrMessageEvent, arg1: str = "", arg2: str = ""
    ):
        """手动查询 Token 使用报告。
        
        用法：
          /token报告                  → 最近 20 轮
          /token报告 轮次 N           → 最近 N 轮
          /token报告 时间 X           → 最近 X 分钟
          /token报告 N                → 最近 N 轮（简写）
        """
        bot_id, _, _ = self._parse_umo(event.unified_msg_origin)

        mode = "round"
        number = 0
        if arg1:
            if arg1 in {"轮次", "回合"}:
                mode = "round"
                number = self._to_int(arg2, 0)
            elif arg1 in {"时间", "分钟"}:
                mode = "time"
                number = self._to_int(arg2, 0)
            else:
                n = self._to_int(arg1, -1)
                if n > 0:
                    mode = "round"
                    number = n
                else:
                    yield event.plain_result(
                        "用法: /token报告 轮次 N | /token报告 时间 X | /token报告 N"
                    )
                    return

        # 默认值：轮次模式 20 轮，时间模式 120 分钟
        if number <= 0:
            number = 20 if mode == "round" else 120

        async with self._lock:
            bot = self._bots.get(bot_id)
            if not bot:
                yield event.plain_result("当前暂无可用统计数据。")
                return

            if mode == "round":
                # 取最近 number 个轮次 ID
                round_ids = [rid for rid, _ in bot.round_events][-number:]
                if not round_ids:
                    yield event.plain_result("当前暂无可用统计数据。")
                    return
                round_set = set(round_ids)
                language_records = [
                    r for r in bot.language_records if r.round_id in round_set
                ]
                speech_records = [
                    r for r in bot.speech_records if r.round_id in round_set
                ]
                image_records = [
                    r for r in bot.image_records if r.round_id in round_set
                ]
            else:
                # 取最近 number 分钟内的记录
                cutoff = datetime.now(timezone.utc) - timedelta(minutes=number)
                language_records = [r for r in bot.language_records if r.ts >= cutoff]
                speech_records = [r for r in bot.speech_records if r.ts >= cutoff]
                image_records = [r for r in bot.image_records if r.ts >= cutoff]

            if not language_records and not speech_records and not image_records:
                yield event.plain_result("当前暂无可用统计数据。")
                return

            report = self._build_report(
                bot_id=bot_id,
                language_records=language_records,
                speech_records=speech_records,
                image_records=image_records,
                include_rounds=True,
            )

        yield event.plain_result(report)

    @filter.command("usage_report_reset")
    async def usage_report_reset(self, event: AstrMessageEvent):
        """重置当前 Bot 的全部统计数据（清空内存中的所有记录）。"""
        bot_id, _, _ = self._parse_umo(event.unified_msg_origin)
        async with self._lock:
            if bot_id in self._bots:
                self._bots[bot_id] = BotState()
        yield event.plain_result("已重置当前 Bot 统计。")

    # ──────────────────────────────────────────────
    # 报告构建
    # ──────────────────────────────────────────────

    def _build_report(
        self,
        bot_id: str,
        language_records: list[LanguageCallRecord],
        speech_records: list[SpeechGenRecord],
        image_records: list[ImageGenRecord],
        include_rounds: bool,
    ) -> str:
        """根据传入的记录列表构建文本格式的报告字符串。
        
        报告按会话分组，每个会话内再按 Provider 分组统计。
        """
        lines: list[str] = [f"📊token使用报告[{bot_id}]"]

        # 收集所有涉及的会话 ID（去重并排序）
        all_sessions = sorted(
            {
                *[r.session_id for r in language_records],
                *[r.session_id for r in speech_records],
                *[r.session_id for r in image_records],
            }
        )

        for session_id in all_sessions:
            lang_arr = [r for r in language_records if r.session_id == session_id]
            speech_arr = [r for r in speech_records if r.session_id == session_id]
            image_arr = [r for r in image_records if r.session_id == session_id]
            kind = (lang_arr or speech_arr or image_arr)[0].chat_kind

            # 统计本会话涉及的有效对话轮次数（round_id > 0 的去重集合）
            round_ids = {
                *[r.round_id for r in lang_arr if r.round_id > 0],
                *[r.round_id for r in speech_arr if r.round_id > 0],
                *[r.round_id for r in image_arr if r.round_id > 0],
            }

            total_calls = len(lang_arr) + len(speech_arr) + len(image_arr)
            p_sum = sum(x.prompt_tokens for x in lang_arr)
            c_sum = sum(x.completion_tokens for x in lang_arr)
            o_sum = sum(x.other_tokens for x in lang_arr)
            t_sum = sum(x.total_tokens for x in lang_arr)

            lines.append(f"🏷️[{session_id}][{self._chat_kind_label(kind)}]")
            if include_rounds:
                lines.append(f"      期间有效对话轮次：{len(round_ids)}次")
            lines.append(f"      期间模型调用总次数：{total_calls}次")
            lines.append(
                f"      期间Token使用总量：{self._fmt_short(t_sum)} "
                f"[输入{self._fmt_short(p_sum)}/输出{self._fmt_short(c_sum)}/其他{self._fmt_short(o_sum)}]"
            )
            lines.append("      调用模型名称")

            # ── 语言模型按 Provider 聚合 ──
            language_map: dict[str, LanguageStats] = {}
            for item in lang_arr:
                stats = language_map.setdefault(item.provider_id, LanguageStats())
                stats.calls += 1
                stats.prompt_tokens += item.prompt_tokens
                stats.completion_tokens += item.completion_tokens
                stats.other_tokens += item.other_tokens
                stats.total_tokens += item.total_tokens
                stats.image_reads += item.image_reads
                stats.audio_reads += item.audio_reads
                stats.max_single_total = max(stats.max_single_total, item.total_tokens)

            # 按总 Token 降序排列，相同则按调用次数降序，再按 provider_id 字母序
            for provider_id, stats in sorted(
                language_map.items(),
                key=lambda x: (-x[1].total_tokens, -x[1].calls, x[0]),
            ):
                avg = int(stats.total_tokens / stats.calls) if stats.calls > 0 else 0
                parts = [
                    f"输入{self._fmt_short(stats.prompt_tokens)}",
                    f"输出{self._fmt_short(stats.completion_tokens)}",
                    f"其他{self._fmt_short(stats.other_tokens)}",
                    f"合计{self._fmt_short(stats.total_tokens)}",
                    f"调用次数{stats.calls}",
                    f"平均调用{self._fmt_short(avg)}",
                    f"最高单次{self._fmt_short(stats.max_single_total)}",
                ]
                if stats.image_reads > 0:
                    parts.append(f"读取图片{self._fmt_short(stats.image_reads)}")
                if stats.audio_reads > 0:
                    parts.append(f"读取音频{self._fmt_short(stats.audio_reads)}")
                lines.append(f"           -{provider_id}[语言]：")
                lines.append(f"               [{'/'.join(parts)}]")

            # ── 语音生成按 Provider 聚合 ──
            speech_map: dict[str, dict[str, float | int]] = {}
            for item in speech_arr:
                e = speech_map.setdefault(
                    item.provider_id, {"chars": 0, "duration_seconds": 0.0, "tokens": 0}
                )
                e["chars"] += item.chars
                e["duration_seconds"] += item.duration_seconds
                e["tokens"] += item.tokens
            for provider_id, e in sorted(speech_map.items(), key=lambda x: x[0]):
                parts = [
                    f"字符数{self._fmt_short(int(e['chars']))}",
                    f"总时长{self._fmt_duration(float(e['duration_seconds']))}",
                ]
                if int(e["tokens"]) > 0:
                    parts.append(f"使用token{self._fmt_short(int(e['tokens']))}")
                lines.append(f"           -{provider_id}[语音生成]：")
                lines.append(f"               [{'/'.join(parts)}]")

            # ── 图片生成按 Provider 聚合 ──
            image_map: dict[str, int] = {}
            for item in image_arr:
                image_map[item.provider_id] = (
                    image_map.get(item.provider_id, 0) + item.image_count
                )
            for provider_id, count in sorted(image_map.items(), key=lambda x: x[0]):
                lines.append(f"           -{provider_id}[图片生成]：")
                lines.append(f"               [图片张数{self._fmt_short(count)}]")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 媒体记录捕获（语音/图片生成）
    # ──────────────────────────────────────────────

    def _capture_generated_media(
        self, event: AstrMessageEvent
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """从已发送消息的结果链中提取语音和图片生成信息。
        
        由于 AstrBot 各适配器的消息组件格式不统一，采用反射+关键词匹配的
        启发式方法检测媒体类型，健壮性优先于精确性。
        """
        speech_items: list[dict[str, Any]] = []
        image_items: list[dict[str, Any]] = []
        try:
            result = event.get_result()
            chain = getattr(result, "chain", None) or []
        except Exception:
            return speech_items, image_items

        for comp in chain:
            flat = self._flatten_kv(comp)
            kind = self._detect_media_kind(comp, flat)
            if not kind:
                continue
            provider_id = self._extract_media_provider_id(flat)
            if kind == "speech":
                speech_items.append(
                    {
                        "provider_id": provider_id or "unknown_tts",
                        "chars": self._extract_tts_chars(flat),
                        "duration_seconds": self._extract_duration_seconds(flat),
                        "tokens": self._extract_media_tokens(flat),
                    }
                )
            elif kind == "image":
                image_items.append(
                    {
                        "provider_id": provider_id or "unknown_image",
                        "image_count": self._extract_image_count(comp, flat),
                    }
                )
        return speech_items, image_items

    def _detect_media_kind(self, comp: Any, flat: list[tuple[str, str]]) -> str:
        """通过类名和属性关键词启发式检测消息组件的媒体类型。
        
        返回 "speech"、"image" 或空字符串（未知/非媒体）。
        """
        blob = " ".join([f"{k}:{v}".lower() for k, v in flat])
        cname = comp.__class__.__name__.lower()

        # 优先通过类名判断
        if any(k in cname for k in ("record", "audio", "voice", "speech", "tts")):
            return "speech"
        # 其次通过属性值中的文件扩展名或关键词判断
        if any(
            k in blob
            for k in (
                ".wav", ".mp3", ".ogg", ".m4a", ".flac", ".aac", ".opus",
                "text_to_speech", "tts",
            )
        ):
            return "speech"
        if any(k in cname for k in ("image", "img", "picture")):
            return "image"
        if any(
            k in blob
            for k in (
                ".png", ".jpg", ".jpeg", ".webp",
                "image_url", "image_path", "image_result",
            )
        ):
            return "image"
        return ""

    def _extract_media_provider_id(self, flat: list[tuple[str, str]]) -> str:
        """从消息组件的属性中提取 Provider ID。
        
        优先使用标准 Provider 字段，其次从文件名中提取，最后使用关键词兜底。
        """
        provider = self._extract_provider(flat)
        if provider:
            return self._normalize_provider_id(provider, fallback="")

        blob = " ".join([f"{k}:{v}".lower() for k, v in flat])
        # 尝试从文件名中提取 Provider 标识，例如：
        #   openai_tts_api_xxx.wav → openai_tts
        #   dashscope_tts_xxx.wav → dashscope_tts
        m = re.search(
            r"([a-z0-9_]+(?:tts|image|stt)[a-z0-9_]*)_(?:api_)?[a-z0-9-]+\.(wav|mp3|ogg|m4a|flac|aac|opus|png|jpg|jpeg|webp)",
            blob,
        )
        if m:
            raw = m.group(1)
            raw = raw.replace("_api", "")
            return raw

        # 关键词兜底
        if "openai_tts" in blob:
            return "openai_tts"
        if "dashscope_tts" in blob:
            return "dashscope_tts"
        return ""

    def _extract_tts_chars(self, flat: list[tuple[str, str]]) -> int:
        """从 TTS 组件属性中提取输入文字字符数。"""
        best = 0
        for key, value in flat:
            if key.lower() in {"text", "content", "tts_text", "speech_text", "input_text"}:
                best = max(best, len(str(value or "").strip()))
        return best

    def _extract_duration_seconds(self, flat: list[tuple[str, str]]) -> float:
        """从媒体组件属性中提取音频时长（秒）。"""
        for key, value in flat:
            if key.lower() in {
                "duration", "duration_seconds", "audio_duration", "seconds", "voice_duration",
            }:
                try:
                    return max(0.0, float(value))
                except Exception:
                    continue
        return 0.0

    def _extract_media_tokens(self, flat: list[tuple[str, str]]) -> int:
        """从媒体组件属性中提取 Token 消耗数（部分 TTS 服务提供此信息）。"""
        for key, value in flat:
            if "token" in key.lower():
                try:
                    return max(0, int(float(value)))
                except Exception:
                    continue
        return 0

    def _extract_image_count(self, comp: Any, flat: list[tuple[str, str]]) -> int:
        """推断本次图片生成的张数。
        
        优先读取列表类属性的长度，其次读取 count/n 等明确字段，最后默认为 1。
        """
        for attr in ("images", "image_urls", "urls", "results"):
            try:
                v = getattr(comp, attr, None)
                if isinstance(v, list) and len(v) > 0:
                    return len(v)
            except Exception:
                continue
        for key, value in flat:
            if key.lower() in {"count", "image_count", "num_images", "n"}:
                try:
                    n = int(float(value))
                    if n > 0:
                        return n
                except Exception:
                    continue
        return 1

    def _count_explicit_assets(self, obj: Any, kind: str) -> int:
        """递归统计请求对象中明确标注为图片或音频的输入字段数量。
        
        采用白名单键名匹配，避免误把模型名称、提示词等字符串计入。
        深度限制为 6 层，防止循环引用或过深结构导致性能问题。
        """
        image_keys = {
            "image", "images", "image_url", "image_urls",
            "image_path", "image_paths", "image_file", "image_files",
        }
        audio_keys = {
            "audio", "audios", "audio_url", "audio_urls",
            "audio_path", "audio_paths", "audio_file", "audio_files",
            "speech", "voice",
        }
        target_keys = image_keys if kind == "image" else audio_keys

        count = 0

        def add_count(v: Any):
            nonlocal count
            if isinstance(v, str) and v.strip():
                count += 1
            elif isinstance(v, list):
                count += len([x for x in v if x is not None and str(x).strip() != ""])
            elif v is not None:
                count += 1

        def walk(x: Any, depth: int = 0):
            if depth > 6 or x is None:
                return
            if isinstance(x, dict):
                for k, v in x.items():
                    key = str(k).lower().strip()
                    if key in target_keys:
                        add_count(v)
                    walk(v, depth + 1)
                return
            if isinstance(x, (list, tuple, set)):
                for item in x:
                    walk(item, depth + 1)
                return
            if isinstance(x, (str, int, float, bool)):
                return
            try:
                attrs = vars(x)
            except Exception:
                attrs = {}
            if attrs:
                walk(attrs, depth + 1)

        walk(obj, 0)
        return count

    # ──────────────────────────────────────────────
    # 报告目标解析
    # ──────────────────────────────────────────────

    def _normalize_provider_id(self, provider: str, fallback: str) -> str:
        """标准化 Provider ID。
        
        常见格式：
          · "group/model"  → 取 "/" 前的 group 部分
          · "provider_id"  → 直接使用
        """
        p = str(provider or "").strip()
        if not p:
            return fallback
        if "/" in p:
            return p.split("/")[0] or fallback
        return p

    def _resolve_report_targets(self, bot: BotState, bot_id: str) -> list[str]:
        """将插件配置的报告目标列表解析为可发送消息的 UMO 字符串列表。
        
        支持以下目标格式（在 report_target_session_ids 配置中填写）：
          · "admin"              → 发送给 AstrBot 全局管理员（admins_id 中的所有成员）
          · "xxx:yyy:zzz"        → 完整 UMO 字符串，直接使用
          · "<session_id>"       → 在 observed_session_map 中查找，找不到则构造私聊 UMO 候选
        """
        raw = self.config.get("report_target_session_ids", ["admin"])
        items = self._normalize_str_list(raw) or ["admin"]
        resolved: list[str] = []

        for item in items:
            if item.lower() == "admin":
                # 从 AstrBot 全局配置中读取管理员 ID 列表
                # 管理员在 AstrBot WebUI → 设置 → 其他设置 → 管理员 ID 列表 中配置
                # 注意：此处的 session_id 与平台用户 ID 对应，与群组权限无关
                try:
                    global_cfg = self.context.get_config()
                    admins_id: list[str] = [
                        str(a).strip()
                        for a in (global_cfg.get("admins_id") or [])
                        if str(a).strip()
                    ]
                except Exception:
                    admins_id = []

                for sid in admins_id:
                    umo = bot.observed_session_map.get(sid, "")
                    if umo:
                        resolved.append(umo)
                continue

            # 完整 UMO（包含两个及以上冒号）直接使用
            if item.count(":") >= 2:
                resolved.append(item)
                continue

            # 尝试从已观察到的会话中查找
            umo = bot.observed_session_map.get(item, "")
            if umo:
                resolved.append(umo)
                continue

            # 兜底：用户只填了 session_id（如 QQ 号），尝试构造常见私聊 UMO
            # 不同适配器私聊消息类型名称不同（FriendMessage / PrivateMessage），各构造一个候选
            resolved.append(f"{bot_id}:FriendMessage:{item}")
            resolved.append(f"{bot_id}:PrivateMessage:{item}")

        # 去重，保持顺序
        out: list[str] = []
        seen = set()
        for umo in resolved:
            if umo in seen:
                continue
            seen.add(umo)
            out.append(umo)
        return out

    # ──────────────────────────────────────────────
    # 内部工具方法
    # ──────────────────────────────────────────────

    def _prune_history(self, bot: BotState):
        """清理 3 天前的历史记录，防止长时间运行后内存无限增长。"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=3)
        bot.language_records = [r for r in bot.language_records if r.ts >= cutoff]
        bot.speech_records = [r for r in bot.speech_records if r.ts >= cutoff]
        bot.image_records = [r for r in bot.image_records if r.ts >= cutoff]
        bot.round_events = [(rid, ts) for rid, ts in bot.round_events if ts >= cutoff]

    def _ensure_bot(self, bot_id: str) -> BotState:
        """获取或创建指定 bot_id 的 BotState。"""
        if bot_id not in self._bots:
            self._bots[bot_id] = BotState()
        return self._bots[bot_id]

    def _ensure_session(self, bot: BotState, session_id: str) -> SessionState:
        """获取或创建指定 session_id 的 SessionState。"""
        if session_id not in bot.sessions:
            bot.sessions[session_id] = SessionState()
        return bot.sessions[session_id]

    def _is_monitored(self, umo: str, session_id: str) -> bool:
        """判断指定会话是否在监控范围内。
        
        若 monitor_session_ids 配置为空列表，则监控所有会话。
        否则仅监控列表中明确列出的 session_id 或完整 UMO。
        """
        items = self._normalize_str_list(self.config.get("monitor_session_ids", []))
        if not items:
            return True
        return (session_id in items) or (umo in items)

    def _is_chat_kind_enabled(self, chat_kind: str) -> bool:
        """根据配置判断指定类型的会话（群聊/私聊）是否启用监控。"""
        if chat_kind == "group":
            return bool(self.config.get("enable_group_monitor", False))
        return bool(self.config.get("enable_private_monitor", True))

    def _normalize_str_list(self, value: Any) -> list[str]:
        """将任意类型的值规范化为非空字符串列表。"""
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _to_int(self, value: Any, default: int) -> int:
        """安全转换为整数，失败时返回默认值。"""
        try:
            return int(value)
        except Exception:
            return default

    def _parse_optional_positive(self, value: Any, max_allowed: int) -> int | None:
        """解析可选的正整数配置项。
        
        返回 None 表示配置为空（禁用该触发方式）。
        返回值被限制在 [1, max_allowed] 范围内。
        """
        text = str(value).strip() if value is not None else ""
        if text == "":
            return None
        try:
            n = int(text)
        except Exception:
            return None
        if n < 1:
            return None
        return min(n, max_allowed)

    def _parse_umo(self, umo: str) -> tuple[str, str, str]:
        """将 unified_msg_origin 字符串解析为 (bot_id, msg_type, session_id) 三元组。
        
        UMO 格式："{bot_id}:{msg_type}:{session_id}"
        例如：  "aiocqhttp:GroupMessage:123456789"
        """
        parts = str(umo or "").split(":", 2)
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        if len(parts) == 2:
            return parts[0], parts[1], ""
        if len(parts) == 1:
            return parts[0], "", ""
        return "", "", ""

    def _chat_kind(self, msg_type: str) -> str:
        """根据消息类型字符串判断会话类型。
        
        凡 msg_type 中包含 "group"（大小写不敏感）的均视为群聊，其余视为私聊。
        """
        return "group" if "group" in str(msg_type or "").lower() else "private"

    def _chat_kind_label(self, chat_kind: str) -> str:
        """将会话类型转为中文标签。"""
        return "群聊" if chat_kind == "group" else "私聊"

    async def _safe_get_provider(self, umo: str) -> str:
        """安全地获取当前会话使用的 Provider ID，失败时返回空字符串。"""
        try:
            return await self.context.get_current_chat_provider_id(umo)
        except Exception:
            return ""

    def _is_admin_event(self, event: AstrMessageEvent) -> bool:
        """判断消息发送者是否为 AstrBot 全局管理员。
        
        以 AstrBot 全局配置中的 admins_id 列表为唯一依据（与群组内权限角色无关）。
        管理员在 AstrBot WebUI → 设置 → 其他设置 → 管理员 ID 列表 中配置。
        """
        try:
            sender_id = str(event.message_obj.sender.user_id).strip()
            global_cfg = self.context.get_config()
            admins_id = [str(a).strip() for a in (global_cfg.get("admins_id") or [])]
            return sender_id in admins_id
        except Exception:
            return False

    def _should_trigger_round(self, bot: BotState, chat_kind: str) -> bool:
        """判断当前轮次数是否达到轮次触发阈值。"""
        if chat_kind == "group":
            threshold = self._parse_optional_positive(
                self.config.get("group_round_trigger", ""), 1000
            )
            return (
                self._is_chat_kind_enabled("group")
                and threshold is not None
                and bot.auto_rounds_group >= threshold
            )
        threshold = self._parse_optional_positive(
            self.config.get("private_round_trigger", ""), 1000
        )
        return (
            self._is_chat_kind_enabled("private")
            and threshold is not None
            and bot.auto_rounds_private >= threshold
        )

    # ──────────────────────────────────────────────
    # Token 提取
    # ──────────────────────────────────────────────

    def _extract_tokens(self, resp: Any) -> tuple[int, int, int]:
        """从 LLM 响应对象中提取 (prompt_tokens, completion_tokens, total_tokens)。
        
        按优先级尝试三种读取方式：
          1. 标准 usage 属性（OpenAI 兼容格式）
          2. usage_metadata 属性（Gemini 格式）
          3. 递归平铺所有属性后按关键词匹配（兜底）
        """
        usage = getattr(resp, "usage", None)
        prompt = self._read_usage(usage, "prompt_tokens", "input_tokens")
        completion = self._read_usage(usage, "completion_tokens", "output_tokens")
        total = self._read_usage(usage, "total_tokens")

        # Gemini 等使用 usage_metadata 字段
        if prompt == 0 and completion == 0 and total == 0:
            usage_meta = getattr(resp, "usage_metadata", None)
            prompt = self._read_usage(
                usage_meta, "prompt_token_count", "input_token_count"
            )
            completion = self._read_usage(
                usage_meta,
                "candidates_token_count",
                "completion_token_count",
                "output_token_count",
            )
            total = self._read_usage(usage_meta, "total_token_count")

        # 最后兜底：平铺所有属性后按关键词查找
        if prompt == 0 and completion == 0 and total == 0:
            flat = self._flatten_kv(resp)
            prompt = self._find_int(
                flat,
                {"prompt_tokens", "input_tokens", "prompt_token_count", "input_token_count"},
            )
            completion = self._find_int(
                flat,
                {
                    "completion_tokens", "output_tokens", "candidates_token_count",
                    "completion_token_count", "output_token_count",
                },
            )
            total = self._find_int(flat, {"total_tokens", "total_token_count"})

        # 若总量字段为 0，则用 prompt+completion 推算
        if total == 0:
            total = prompt + completion
        return prompt, completion, total

    def _read_usage(self, usage: Any, *keys: str) -> int:
        """从 usage 对象（dict 或 object）中按优先级读取整数字段。"""
        for key in keys:
            if usage is None:
                continue
            value = (
                usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
            )
            try:
                if value is not None:
                    return int(float(value))
            except Exception:
                continue
        return 0

    def _flatten_kv(self, obj: Any, max_depth: int = 5) -> list[tuple[str, str]]:
        """将任意嵌套对象递归平铺为 (key, value) 字符串元组列表。
        
        用于在不知道对象具体类型的情况下，通过关键词匹配提取所需字段。
        支持 dict、list、dataclass、Pydantic model 等常见结构。
        通过 seen 集合避免循环引用导致的无限递归。
        """
        out: list[tuple[str, str]] = []
        seen: set[int] = set()

        def walk(x: Any, depth: int):
            if depth > max_depth:
                return
            oid = id(x)
            if oid in seen:
                return
            seen.add(oid)
            if x is None or isinstance(x, (str, int, float, bool)):
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
            try:
                attrs = vars(x)
            except Exception:
                attrs = {}
            if not attrs:
                # 尝试 Pydantic model_dump 或旧版 dict()
                for meth in ("model_dump", "dict"):
                    fn = getattr(x, meth, None)
                    if callable(fn):
                        try:
                            data = fn()
                            if isinstance(data, dict):
                                attrs = data
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
        """从平铺属性列表中提取 Provider ID 字段。"""
        keys = {
            "chat_provider_id", "provider_id", "provider",
            "llm_provider_id", "provider_name",
        }
        for key, value in flat:
            if key in keys and value:
                return value
        return ""

    def _find_int(self, flat: list[tuple[str, str]], keys: set[str]) -> int:
        """在平铺属性列表中按关键词集合查找并返回第一个整数值。"""
        for key, value in flat:
            if key not in keys:
                continue
            try:
                return int(float(value))
            except Exception:
                continue
        return 0

    # ──────────────────────────────────────────────
    # 格式化工具
    # ──────────────────────────────────────────────

    def _fmt_short(self, value: int) -> str:
        """将整数格式化为易读的短字符串。
        
        规则：
          · ≥ 1,000,000  → x.xxm（保留两位小数）
          · ≥ 10,000     → x.xk（保留一位小数）
          · 其他         → 原始整数字符串
        """
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

    def _fmt_duration(self, seconds: float) -> str:
        """将秒数格式化为带单位的字符串（保留一位小数）。"""
        if seconds <= 0:
            return "0s"
        return f"{seconds:.1f}s"
