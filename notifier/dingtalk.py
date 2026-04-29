# notifier/dingtalk.py
import os
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests
import re
from datetime import datetime, timezone, timedelta
from typing import Optional
from utils.logger import logger


# ===================== 基础推送 =====================
def send_dingtalk_message(content: str, title: str = "策略推送") -> bool:
    if not content or not content.strip():
        logger.error(f"钉钉推送内容为空，已跳过: title={title}")
        return False

    webhook = os.getenv("DINGTALK_WEBHOOK_URL", "")
    secret = os.getenv("DINGTALK_SECRET", "")
    if not webhook:
        logger.error("未配置钉钉 Webhook")
        return False

    ts = str(round(time.time() * 1000))
    if secret and secret.lower() != "none":
        try:
            sign_str = f"{ts}\n{secret}"
            signature = base64.b64encode(
                hmac.new(secret.encode(), sign_str.encode(), hashlib.sha256).digest()
            ).decode()
            sign = urllib.parse.quote(signature, safe="")
            url_parts = list(urllib.parse.urlparse(webhook))
            query = dict(urllib.parse.parse_qsl(url_parts[4]))
            query.update({"timestamp": ts, "sign": sign})
            url_parts[4] = urllib.parse.urlencode(query)
            webhook = urllib.parse.urlunparse(url_parts)
        except Exception as e:
            logger.error(f"钉钉签名生成失败: {e}")
            return False

    try:
        payload = {"msgtype": "markdown", "markdown": {"title": title, "text": content}}
        resp = requests.post(webhook, json=payload, timeout=10)
        resp_data = resp.json()
        if resp_data.get("errcode") == 0:
            logger.info("钉钉推送成功")
            return True
        logger.error(f"钉钉失败: {resp_data}")
        return False
    except Exception as e:
        logger.error(f"钉钉异常: {e}")
        return False


# ===================== 文本清理 =====================
def _safe_code_block(text: str) -> str:
    if not text:
        return ""
    return text.replace("```", "'''")


# ===================== 裁决文本解析（强化版） =====================
def _parse_judge_execution(text: str) -> dict:
    """
    强制从裁决文本中提取最终判决和执行指令。
    提取失败时返回空字典，由上层决定默认值。
    """
    result = {}
    if not text:
        return result

    # ----- 1. 提取最终判决（支持跨行） -----
    lines = text.split('\n')
    verdict_raw = ""
    for i, line in enumerate(lines):
        m = re.search(r'(?:📌\s*)?最终判决[：:]\s*(.*)', line)
        if m:
            verdict_raw = m.group(1).strip()
            if not verdict_raw:  # 冒号后为空，取下一非空行
                for j in range(i + 1, len(lines)):
                    nxt = lines[j].strip()
                    if nxt:
                        verdict_raw = nxt
                        break
            break

    if not verdict_raw:
        # 尝试模糊匹配整段
        m2 = re.search(r'最终判决[：:]\s*([^\n]*)', text)
        if m2:
            verdict_raw = m2.group(1).strip()
    if verdict_raw:
        result["verdict_raw"] = verdict_raw
        result["verdict"] = "推翻" if "推翻" in verdict_raw else "维持原判"
    else:
        result["verdict_raw"] = ""
        result["verdict"] = ""

    # ----- 2. 定位执行指令块 -----
    exec_start = re.search(r'🎯\s*执行指令', text)
    if not exec_start:
        return result
    exec_text = text[exec_start.start():]

    # ----- 2.1 方向 -----
    m_dir = re.search(r'方向[：:]\s*([^\n，,]+)', exec_text)
    if m_dir:
        raw_dir = m_dir.group(1).strip()
        if "做多" in raw_dir or "多" in raw_dir:
            result["direction"] = "long"
        elif "做空" in raw_dir or "空" in raw_dir:
            result["direction"] = "short"
        elif "观望" in raw_dir:
            result["direction"] = "neutral"

    if "direction" not in result:
        m_dir2 = re.search(r'(做多|做空|观望)', exec_text)
        if m_dir2:
            d = m_dir2.group(1)
            result["direction"] = {"做多": "long", "做空": "short", "观望": "neutral"}[d]

    # ----- 2.2 仓位 -----
    m_pos = re.search(r'仓位[：:]\s*([^\n，,]+)', exec_text)
    if m_pos:
        raw_pos = m_pos.group(1).strip()
        if "轻" in raw_pos:
            result["position_size"] = "light"
        elif "重" in raw_pos:
            result["position_size"] = "heavy"
        elif "中" in raw_pos:
            result["position_size"] = "medium"
        elif "无" in raw_pos:
            result["position_size"] = "none"

    # ----- 2.3 入场区间 -----
    m_entry = re.search(r'入场区间[：:]\s*([\d.,]+)\s*[-~至]+?\s*([\d.,]+)', exec_text)
    if m_entry:
        try:
            result["entry_low"] = float(m_entry.group(1).replace(",", ""))
            result["entry_high"] = float(m_entry.group(2).replace(",", ""))
        except ValueError:
            pass

    # ----- 2.4 止损 -----
    m_sl = re.search(r'止损[：:]\s*([\d.,]+)', exec_text)
    if m_sl:
        try:
            result["stop_loss"] = float(m_sl.group(1).replace(",", ""))
        except ValueError:
            pass

    # ----- 2.5 止盈 -----
    m_tp = re.search(r'止盈[：:]\s*([\d.,]+)', exec_text)
    if m_tp:
        try:
            result["take_profit"] = float(m_tp.group(1).replace(",", ""))
        except ValueError:
            pass

    return result


# ===================== 发送（无分块，直接完整发送） =====================
def _send_full_message(body: str, title: str) -> None:
    """直接发送完整消息，记录日志，并确保内容非空"""
    if not body or not body.strip():
        logger.error(f"消息体为空，取消发送: title={title}")
        return
    if len(body) > 4000:
        logger.warning(f"消息长度 {len(body)} 超过4000字符，钉钉可能会截断显示")
    if not send_dingtalk_message(body, title):
        logger.error(f"发送失败: {title}")


# ===================== 消息构建 =====================
def format_strategy_message(symbol: str, strategy: dict, data: dict) -> None:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    direction = strategy.get("direction", "neutral")
    pos_size = strategy.get("position_size", "none")
    conf = strategy.get("confidence", "medium")
    entry_low = strategy.get("entry_price_low", 0) or 0
    entry_high = strategy.get("entry_price_high", 0) or 0
    stop_loss = strategy.get("stop_loss", 0) or 0
    take_profit = strategy.get("take_profit", 0) or 0
    current = (data.get("mark_price", 0) or 0) if data else 0

    dir_icon = {"long": "🟢", "short": "🔴", "neutral": "⚪"}.get(direction, "⚪")
    dir_text = {"long": "做多", "short": "做空", "neutral": "观望"}.get(direction, "观望")
    size_text = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}.get(pos_size, "?")
    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "?")

    header = f"### 🧠 首席交易员 · 提交审计 | {symbol} | {dir_icon} {dir_text} | {size_text} | 置信度{conf_icon} | {now}"
    price_block = (
        f"现价 {current:.0f}  |  "
        f"入场 {entry_low:.0f}-{entry_high:.0f}  |  "
        f"止损 {stop_loss:.0f}  |  "
        f"止盈 {take_profit:.0f}"
    )

    full_reasoning = strategy.get("reasoning", "")
    if not full_reasoning:
        full_reasoning = "无推演内容"

    body = (
        f"{header}\n"
        f"{price_block}\n\n"
        f"**推演过程**\n"
        f"```\n{_safe_code_block(full_reasoning)}\n```"
    )
    _send_full_message(body, f"首席交易员·{symbol}")


def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> None:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    severity = reviewer_report.get("severity_counts", {"高": 0, "中": 0, "低": 0})
    high = severity.get("高", 0)
    medium = severity.get("中", 0)
    low = severity.get("低", 0)

    conclusion = "⛔ 驳回" if high > 0 else ("⚠️ 存疑" if medium > 0 or low > 0 else "✅ 通过")

    header = f"### ⚡ 风控审计官 · 审计完成 | {symbol} | {conclusion} | {now}"
    sev_line = f"严重 {high}  中等 {medium}  轻微 {low}"
    report = reviewer_report.get("full_report", "无审查报告")

    body = (
        f"{header}\n"
        f"{sev_line}\n\n"
        f"**审计报告**\n"
        f"```\n{_safe_code_block(report)}\n```"
    )
    _send_full_message(body, f"风控审计·{symbol}")


def format_final_decision(symbol: str, strategy: dict, judge_result: dict = None, data: dict = None) -> None:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    # 获取裁决全文
    judge_full_text = ""
    if isinstance(judge_result, dict):
        judge_full_text = judge_result.get("content", "") or judge_result.get("full_report", "") or ""
    if not judge_full_text:
        judge_full_text = strategy.get("_judge_reasoning", "")

    # 解析裁决内容
    parsed = _parse_judge_execution(judge_full_text) if judge_full_text else {}

    # ----- 必须从裁决内容中提取的字段，缺失时使用安全默认值 -----
    verdict_raw = parsed.get("verdict_raw", "")
    final_direction = parsed.get("direction", "neutral")
    final_pos_size = parsed.get("position_size", "none")
    final_entry_low = parsed.get("entry_low", 0)
    final_entry_high = parsed.get("entry_high", 0)
    final_stop_loss = parsed.get("stop_loss", 0)
    final_take_profit = parsed.get("take_profit", 0)

    # 如果关键字段完全缺失，尝试用 strategy 兜底（但只在解析完全失败时）
    if not verdict_raw:
        verdict_raw = "维持原判"
        final_direction = strategy.get("direction", final_direction)
        final_pos_size = strategy.get("position_size", final_pos_size)
        final_entry_low = final_entry_low or strategy.get("entry_price_low", 0) or 0
        final_entry_high = final_entry_high or strategy.get("entry_price_high", 0) or 0
        final_stop_loss = final_stop_loss or strategy.get("stop_loss", 0) or 0
        final_take_profit = final_take_profit or strategy.get("take_profit", 0) or 0

    current = (data.get("mark_price", 0) or 0) if data else 0

    verdict_icon = "✅" if "维持" in verdict_raw else "🔄"

    dir_icon = {"long": "🟢", "short": "🔴", "neutral": "⚪"}.get(final_direction, "⚪")
    dir_text = {"long": "做多", "short": "做空", "neutral": "观望"}.get(final_direction, "观望")
    size_text = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}.get(final_pos_size, "?")
    conf_icon = "🟢" if strategy.get("confidence") == "high" else ("🟡" if strategy.get("confidence") == "medium" else "🔴")

    header = f"### 📋 交易委员会 · 最终裁决 | {symbol} | {verdict_icon} {verdict_raw} | {now}"
    decision_line = f"{dir_icon} {dir_text} | {size_text} | 置信度{conf_icon}"
    price_block = (
        f"现价 {current:.1f}  |  "
        f"入场 {final_entry_low:.0f}-{final_entry_high:.0f}  |  "
        f"止损 {final_stop_loss:.0f}  |  "
        f"止盈 {final_take_profit:.0f}"
    )

    if not judge_full_text:
        judge_full_text = "无裁决内容"

    body = (
        f"{header}\n"
        f"{decision_line}\n"
        f"{price_block}\n\n"
        f"**裁决内容**\n"
        f"```\n{_safe_code_block(judge_full_text)}\n```"
    )
    _send_full_message(body, f"最终裁决·{symbol}")


# 兼容旧名
def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> None:
    format_final_decision(symbol, strategy, judge_result, data)
