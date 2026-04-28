import os
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests
import re
from datetime import datetime, timezone, timedelta
from utils.logger import logger


# ========== 基础推送 ==========
def send_dingtalk_message(content: str, title: str = "策略推送") -> bool:
    webhook = os.getenv("DINGTALK_WEBHOOK_URL", "")
    secret = os.getenv("DINGTALK_SECRET", "")
    if not webhook:
        logger.error("未配置钉钉 Webhook")
        return False
    ts = str(round(time.time() * 1000))
    if secret and secret.lower() != "none":
        sign_str = f"{ts}\n{secret}"
        sign = urllib.parse.quote_plus(base64.b64encode(hmac.new(secret.encode(), sign_str.encode(), hashlib.sha256).digest()))
        webhook = f"{webhook}&timestamp={ts}&sign={sign}"
    try:
        payload = {"msgtype": "markdown", "markdown": {"title": title, "text": content}}
        resp = requests.post(webhook, json=payload, timeout=10)
        if resp.json().get("errcode") == 0:
            logger.info("钉钉推送成功")
            return True
        logger.error(f"钉钉失败: {resp.json()}")
        return False
    except Exception as e:
        logger.error(f"钉钉异常: {e}")
        return False


# ========== 长度保护 ==========
def _safe_truncate(text: str, max_len: int = 3500) -> str:
    if len(text) <= max_len:
        return text
    hint = "\n\n... (内容过长已截断，完整信息见运行日志)"
    return text[:max_len - len(hint)] + hint


# ========== 1. 首席交易员初步信号 ==========
def format_strategy_message(symbol: str, strategy: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    direction = strategy.get("direction", "neutral")
    pos_size = strategy.get("position_size", "none")
    conf = strategy.get("confidence", "medium")
    entry_low = strategy.get("entry_price_low", 0)
    entry_high = strategy.get("entry_price_high", 0)
    stop_loss = strategy.get("stop_loss", 0)
    take_profit = strategy.get("take_profit", 0)
    current = data.get("mark_price", 0)

    dir_map = {"long": "🟢 做多", "short": "🔴 做空", "neutral": "⚪ 观望"}
    size_map = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}
    conf_map = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}

    # 每一行末加两个空格确保换行
    line1 = f"**策略信号：{symbol}｜🧠 首席交易员 🔍待审计**  "
    line2 = f"{dir_map.get(direction, '⚪ 观望')} · {size_map.get(pos_size, '未知')} · {conf_map.get(conf, '？')}   {now}  "
    line3 = f"现价：{current:.0f} · 入场：{entry_low:.0f}-{entry_high:.0f} · 止损：{stop_loss:.0f} · 止盈：{take_profit:.0f}  "

    reasoning = strategy.get("reasoning", "无推演过程")
    reasoning = _safe_truncate(reasoning)

    # 拼接：前三行用软换行，然后空一行，接着代码块标题和代码块
    body = (
        f"{line1}\n"
        f"{line2}\n"
        f"{line3}\n\n"
        f"**完整推演过程**\n"
        f"```\n{reasoning}\n```"
    )
    return body


# ========== 2. 风控审计报告 ==========
def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    severity = reviewer_report.get("severity_counts", {"高": 0, "中": 0, "低": 0})
    high = severity.get("高", 0)
    medium = severity.get("中", 0)
    low = severity.get("低", 0)

    if high > 0:
        conclusion = "❌ 驳回"
    elif medium > 0 or low > 0:
        conclusion = "⚠️ 存疑"
    else:
        conclusion = "✅ 通过"

    # 标题行末加两个空格
    line1 = f"**策略信号：{symbol}｜⚡ 风控审计官 📋审计完成**  "
    line2 = f"结论：{conclusion}   {now}  "
    line3 = f"🔴严重 {high}  "
    line4 = f"🟡中等 {medium}  "
    line5 = f"⚪轻微 {low}  "

    report = reviewer_report.get("full_report", "无审查报告")
    report = _safe_truncate(report)

    body = (
        f"{line1}\n"
        f"{line2}\n"
        f"{line3}\n"
        f"{line4}\n"
        f"{line5}\n\n"
        f"**完整审计报告**\n"
        f"```\n{report}\n```"
    )
    return body


# ========== 3. 交易委员会最终裁决 ==========
def format_final_decision(symbol: str, strategy: dict, judge_result: dict = None, data: dict = None) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    direction = strategy.get("direction", "neutral")
    pos_size = strategy.get("position_size", "none")
    conf = strategy.get("confidence", "medium")
    verdict = strategy.get("_review_verdict", "")
    entry_low = strategy.get("entry_price_low", 0)
    entry_high = strategy.get("entry_price_high", 0)
    stop_loss = strategy.get("stop_loss", 0)
    take_profit = strategy.get("take_profit", 0)
    current = data.get("mark_price", 0) if data else 0

    dir_map = {"long": "🟢做多", "short": "🔴做空", "neutral": "⚪观望"}
    size_map = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}
    conf_map = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}

    verdict_map = {
        "维持原判": "✅维持原判",
        "修正参数": "🔧修正参数",
        "降级执行": "⚠️降级执行",
        "推翻改为观望": "🔄推翻→观望",
        "推翻改为反向操作": "🔄推翻·反向"
    }

    # 每行末加两个空格
    line1 = f"**策略信号：{symbol}｜📋 交易委员会 ⚖️最终裁决**  "
    line2 = f"{verdict_map.get(verdict, verdict)} · {dir_map.get(direction, '')} · {size_map.get(pos_size, '')} · {conf_map.get(conf, '')}   {now}  "
    line3 = f"现价：{current:.1f} · 入场：{entry_low:.0f}-{entry_high:.0f} · 止损：{stop_loss:.0f} · 止盈：{take_profit:.0f}  "

    judge_content = (
        strategy.get("_title_line", "") + "\n" +
        strategy.get("_exec_block_raw", "") + "\n" +
        strategy.get("_reasoning_block_raw", "") + "\n" +
        strategy.get("_risk_block_raw", "")
    )
    judge_content = _safe_truncate(judge_content.strip())

    body = (
        f"{line1}\n"
        f"{line2}\n"
        f"{line3}\n\n"
        f"**完整裁决内容**\n"
        f"```\n{judge_content}\n```"
    )
    return body


# ========== 兼容旧版 ==========
def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> str:
    return format_final_decision(symbol, strategy, judge_result, data)
