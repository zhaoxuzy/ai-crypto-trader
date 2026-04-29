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


# ===================== 文本提取 =====================
def _extract_final_step(text: str) -> str:
    if not text:
        return ""
    pattern = r'第七步[：:]\s*[^\n]*\n(.*?)(?=\n第[一二三四五六七八九十]+\s*步|\Z)'
    m = re.search(pattern, text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()
    m = re.search(r'(第七步[：:].*)', text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return text[-2000:].lstrip('\n')


def _safe_code_block(text: str) -> str:
    if not text:
        return ""
    return text.replace("```", "'''")


# ===================== 裁决文本解析 =====================
def _parse_judge_execution(text: str) -> dict:
    result = {}
    if not text:
        return result

    # 1. 提取最终判决
    m = re.search(r'📌\s*最终判决[：:]\s*(.*?)(?:\n|$)', text)
    if not m:
        m = re.search(r'最终判决[：:]\s*(.*?)(?:\n|$)', text)
    if m:
        raw = m.group(1).strip()
        result["verdict"] = "推翻" if "推翻" in raw else "维持原判"
    else:
        result["verdict"] = "维持原判"   # 默认

    # 2. 定位执行指令块
    exec_start = re.search(r'🎯\s*执行指令', text)
    if not exec_start:
        return result
    exec_text = text[exec_start.start():]

    # 辅助提取
    def _extract(pattern, target, default=""):
        m = re.search(pattern, target)
        return m.group(1).strip() if m else default

    result["direction"] = _extract(r'方向[：:]\s*([做多空观望]+)', exec_text)
    pos_raw = _extract(r'仓位[：:]\s*([^\n]+)', exec_text)
    if pos_raw:
        pos_raw = pos_raw.split("（")[0].split("(")[0].strip()
    result["position_size"] = pos_raw

    # 入场区间
    entry_match = re.search(r'入场区间[：:]\s*([\d.,]+)\s*[-~至]+\s*([\d.,]+)', exec_text)
    if entry_match:
        try:
            result["entry_low"] = float(entry_match.group(1).replace(",", ""))
            result["entry_high"] = float(entry_match.group(2).replace(",", ""))
        except ValueError:
            pass

    # 止损
    sl_match = re.search(r'止损[：:]\s*([\d.,]+)', exec_text)
    if sl_match:
        try:
            result["stop_loss"] = float(sl_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # 止盈
    tp_match = re.search(r'止盈[：:]\s*([\d.,]+)', exec_text)
    if tp_match:
        try:
            result["take_profit"] = float(tp_match.group(1).replace(",", ""))
        except ValueError:
            pass

    return result


# ===================== 超长拆分发送 =====================
DINGTALK_MAX_CHARS = 4000

def _send_long_with_code_block(body: str, title: str) -> bool:
    if len(body) <= DINGTALK_MAX_CHARS:
        return send_dingtalk_message(body, title)

    code_start = body.find("```")
    if code_start == -1:
        return _send_long_fallback(body, title)

    before = body[:code_start].rstrip()
    after_start = body[code_start+3:]
    code_end = after_start.find("```")
    if code_end == -1:
        return _send_long_fallback(body, title)

    code_content = after_start[:code_end].strip()
    after_code = after_start[code_end+3:].strip()

    if before:
        if not send_dingtalk_message(before + "\n\n*（详细内容见下一条）*", title):
            return False
        time.sleep(0.6)

    return _send_codeblock_split(code_content, after_code, title)


def _send_codeblock_split(code_content: str, after_text: str, title: str) -> bool:
    page_footer = "\n\n*（推演 {}/{}）*"
    max_chunk = DINGTALK_MAX_CHARS - len(page_footer.format(99,99)) - 10

    full_block = f"```\n{code_content}\n```"
    if after_text:
        full_block += f"\n{after_text}"
    if len(full_block) <= DINGTALK_MAX_CHARS:
        return send_dingtalk_message(full_block, title)

    chunks = []
    remaining = code_content
    while remaining:
        if len(remaining) <= max_chunk:
            chunks.append(remaining)
            break
        cut_pos = remaining.rfind('\n', 0, max_chunk)
        if cut_pos == -1 or cut_pos < max_chunk//2:
            cut_pos = remaining.rfind(' ', 0, max_chunk)
        if cut_pos == -1 or cut_pos < max_chunk//2:
            cut_pos = max_chunk
        chunks.append(remaining[:cut_pos].rstrip())
        remaining = remaining[cut_pos:].lstrip('\n')

    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        msg = f"```\n{chunk}\n```" + page_footer.format(i, total)
        if i == total and after_text:
            msg += f"\n{after_text}"
        if not send_dingtalk_message(msg, f"{title}({i}/{total})"):
            return False
        if i < total:
            time.sleep(0.6)
    return True


def _send_long_fallback(body: str, title: str) -> bool:
    page_footer = "\n\n*（续 {}/{}）*"
    max_chunk = DINGTALK_MAX_CHARS - len(page_footer.format(99,99)) - 10
    chunks = []
    remaining = body
    while remaining:
        if len(remaining) <= max_chunk:
            chunks.append(remaining)
            break
        cut = remaining.rfind('\n', 0, max_chunk)
        if cut == -1:
            cut = max_chunk
        chunk = remaining[:cut].rstrip()
        chunks.append(chunk)
        remaining = remaining[cut:].lstrip('\n')
    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        msg = chunk + page_footer.format(i, total)
        if not send_dingtalk_message(msg, f"{title} ({i}/{total})"):
            return False
        if i < total:
            time.sleep(0.6)
    return True


# ===================== 消息构建 =====================
def format_strategy_message(symbol: str, strategy: dict, data: dict) -> bool:
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

    final_step = _extract_final_step(strategy.get("reasoning", ""))
    if not final_step:
        final_step = "未解析到第七步"

    body = (
        f"{header}\n"
        f"{price_block}\n\n"
        f"**第七步 · 交易计划**\n"
        f"```\n{_safe_code_block(final_step)}\n```"
    )
    return _send_long_with_code_block(body, f"首席交易员·{symbol}")


def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> bool:
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
    return _send_long_with_code_block(body, f"风控审计·{symbol}")


def format_final_decision(symbol: str, strategy: dict, judge_result: dict = None, data: dict = None) -> bool:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    # ----- 获取裁决全文 -----
    judge_full_text = ""
    if isinstance(judge_result, dict):
        judge_full_text = judge_result.get("content", "") or judge_result.get("full_report", "") or ""
    if not judge_full_text:
        judge_full_text = strategy.get("_judge_reasoning", "")

    # ----- 解析执行指令 -----
    parsed = _parse_judge_execution(judge_full_text) if judge_full_text else {}

    # 最终参数优先级：解析出的 > 原始 strategy（fallback）
    final_direction = parsed.get("direction") or strategy.get("direction", "neutral")
    final_pos_size = parsed.get("position_size") or strategy.get("position_size", "none")
    final_entry_low = parsed.get("entry_low")
    if final_entry_low is None:
        final_entry_low = strategy.get("entry_price_low", 0) or 0
    final_entry_high = parsed.get("entry_high")
    if final_entry_high is None:
        final_entry_high = strategy.get("entry_price_high", 0) or 0
    final_stop_loss = parsed.get("stop_loss")
    if final_stop_loss is None:
        final_stop_loss = strategy.get("stop_loss", 0) or 0
    final_take_profit = parsed.get("take_profit")
    if final_take_profit is None:
        final_take_profit = strategy.get("take_profit", 0) or 0

    current = (data.get("mark_price", 0) or 0) if data else 0

    # 裁决判决（仅两种）
    verdict = parsed.get("verdict", "维持原判")
    if verdict not in ("维持原判", "推翻"):
        verdict = "维持原判"

    # 图标与文字
    dir_icon = {"long": "🟢", "short": "🔴", "neutral": "⚪"}.get(final_direction, "⚪")
    dir_text = {"long": "做多", "short": "做空", "neutral": "观望"}.get(final_direction, "观望")
    size_text = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}.get(final_pos_size, "?")
    conf_icon = "🟢" if strategy.get("confidence") == "high" else ("🟡" if strategy.get("confidence") == "medium" else "🔴")

    verdict_text = "✅维持原判" if verdict == "维持原判" else "🔄推翻"
    verdict_icon = "✅" if verdict == "维持原判" else "🔄"

    header = f"### 📋 交易委员会 · 最终裁决 | {symbol} | {verdict_icon} {verdict_text} | {now}"
    decision_line = f"{dir_icon} {dir_text} | {size_text} | 置信度{conf_icon}"
    price_block = (
        f"现价 {current:.1f}  |  "
        f"入场 {final_entry_low:.0f}-{final_entry_high:.0f}  |  "
        f"止损 {final_stop_loss:.0f}  |  "
        f"止盈 {final_take_profit:.0f}"
    )

    # 裁决内容全文作为代码块
    if not judge_full_text:
        judge_full_text = "无裁决内容"

    body = (
        f"{header}\n"
        f"{decision_line}\n"
        f"{price_block}\n\n"
        f"**裁决内容**\n"
        f"```\n{_safe_code_block(judge_full_text)}\n```"
    )
    return _send_long_with_code_block(body, f"最终裁决·{symbol}")


# 兼容旧名
def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> bool:
    return format_final_decision(symbol, strategy, judge_result, data)
