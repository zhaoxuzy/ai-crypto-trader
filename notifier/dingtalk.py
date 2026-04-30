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
    # 更严格的签名判断
    if secret and secret.strip().lower() not in ("", "none"):
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
    """替换连续三个及以上反引号，防止破坏 markdown 代码块"""
    if not text:
        return ""
    return re.sub(r'`{3,}', lambda m: "'" * len(m.group()), text)


# ===================== 裁决文本解析 =====================
def _parse_judge_execution(text: str) -> dict:
    result = {}
    if not text:
        return result

    m = re.search(r'(?:📌\s*)?最终判决[：:]\s*(.*)', text)
    if m:
        verdict_raw = m.group(1).strip()
        if not verdict_raw:
            rest = text[m.end():].strip().split('\n')
            verdict_raw = rest[0].strip() if rest else ""
        result["verdict_raw"] = verdict_raw
        result["verdict"] = "推翻" if "推翻" in verdict_raw else "维持原判"
    else:
        result["verdict_raw"] = ""
        result["verdict"] = ""

    exec_start = re.search(r'🎯\s*执行指令', text)
    if not exec_start:
        return result
    exec_text = text[exec_start.start():]

    # 方向
    m_dir = re.search(r'方向[：:]\s*([^\n，,]+)', exec_text)
    if m_dir:
        raw_dir = m_dir.group(1).strip()
        dir_match = re.search(r'(做多|做空|观望)', raw_dir)
        if dir_match:
            result["direction"] = {"做多": "long", "做空": "short", "观望": "neutral"}[dir_match.group(1)]
    if "direction" not in result:
        m_dir2 = re.search(r'(做多|做空|观望)', exec_text)
        if m_dir2:
            result["direction"] = {"做多": "long", "做空": "short", "观望": "neutral"}[m_dir2.group(1)]

    # 仓位
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

    # 入场区间
    m_entry = re.search(r'入场区间[：:]\s*([\d.,]+)\s*[-~至]+?\s*([\d.,]+)', exec_text)
    if m_entry:
        try:
            result["entry_low"] = float(m_entry.group(1).replace(",", ""))
            result["entry_high"] = float(m_entry.group(2).replace(",", ""))
        except ValueError:
            pass

    # 止损 / 止盈
    for key, regex in [("stop_loss", r'止损[：:]\s*([\d.,]+)'),
                       ("take_profit", r'止盈[：:]\s*([\d.,]+)')]:
        m = re.search(regex, exec_text)
        if m:
            try:
                result[key] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

    return result


# ===================== 消息构建 =====================
def format_strategy_message(symbol: str, strategy: dict, data: dict) -> str:
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

    # ---- 修复1：确保推演内容是完整的多行文本 ----
    full_reasoning = strategy.get("reasoning", "")
    if not full_reasoning:
        # 尝试从其他可能字段获取
        full_reasoning = strategy.get("full_reasoning") or strategy.get("raw") or "无推演内容"
    # 如果内容只有一行且很长，可能是未换行，但保留原样
    if isinstance(full_reasoning, str) and "\n" not in full_reasoning and len(full_reasoning) > 120:
        logger.warning(f"策略推演内容疑似单行大段文本，建议检查 reasoning 字段是否包含换行: {symbol}")

    body = (
        f"{header}\n"
        f"{price_block}\n\n"
        f"**推演过程**\n"
        f"```\n{_safe_code_block(full_reasoning)}\n```"
    )
    return body


def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    # ---- 修复2：智能获取严重/中等/轻微统计 ----
    high = medium = low = 0

    # 方式1：标准 severity_counts 字典
    if "severity_counts" in reviewer_report:
        sev = reviewer_report["severity_counts"]
        high = sev.get("高", sev.get("high", 0))
        medium = sev.get("中", sev.get("medium", 0))
        low = sev.get("低", sev.get("low", 0))
    # 方式2：独立字段
    elif "high" in reviewer_report or "严重" in reviewer_report:
        high = reviewer_report.get("high", reviewer_report.get("严重", 0))
        medium = reviewer_report.get("medium", reviewer_report.get("中等", 0))
        low = reviewer_report.get("low", reviewer_report.get("轻微", 0))
    # 方式3：从 full_report 文本中统计关键词
    else:
        report_text = reviewer_report.get("full_report", "")
        if report_text:
            high = report_text.count("严重")
            medium = report_text.count("中等")
            low = report_text.count("轻微")
            logger.info(f"从审计报告文本中统计: 严重{high} 中等{medium} 轻微{low}")

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
    return body


def format_final_decision(symbol: str, strategy: dict, judge_result: dict = None, data: dict = None) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    judge_full_text = ""
    if isinstance(judge_result, dict):
        judge_full_text = judge_result.get("content", "") or judge_result.get("full_report", "") or ""
    if not judge_full_text:
        judge_full_text = strategy.get("_judge_reasoning", "")

    parsed = _parse_judge_execution(judge_full_text) if judge_full_text else {}

    verdict_raw = parsed.get("verdict_raw", "")
    final_direction = parsed.get("direction", "neutral")
    final_pos_size = parsed.get("position_size", "none")
    final_entry_low = parsed.get("entry_low", 0) or 0
    final_entry_high = parsed.get("entry_high", 0) or 0
    final_stop_loss = parsed.get("stop_loss", 0) or 0
    final_take_profit = parsed.get("take_profit", 0) or 0

    # 回退逻辑：当解析不到有效字段时，继承原策略
    if not verdict_raw:
        verdict_raw = "维持原判"
        final_direction = strategy.get("direction", final_direction)
        final_pos_size = strategy.get("position_size", final_pos_size)
        final_entry_low = final_entry_low or strategy.get("entry_price_low", 0) or 0
        final_entry_high = final_entry_high or strategy.get("entry_price_high", 0) or 0
        final_stop_loss = final_stop_loss or strategy.get("stop_loss", 0) or 0
        final_take_profit = final_take_profit or strategy.get("take_profit", 0) or 0
    else:
    # 有判决但部分字段缺失时，温和继承原策略内容
    if "direction" not in parsed or parsed.get("direction") is None:
        final_direction = strategy.get("direction", final_direction)
    
    # 仓位：仅当裁决未给出有效仓位时才继承
    if final_pos_size == "none" and strategy.get("position_size") not in (None, "none"):
        final_pos_size = strategy["position_size"]
    
    # 价格字段：裁决未给出时继承原策略
    if not final_entry_low:
        final_entry_low = strategy.get("entry_price_low", 0) or 0
    if not final_entry_high:
        final_entry_high = strategy.get("entry_price_high", 0) or 0
    if not final_stop_loss:
        final_stop_loss = strategy.get("stop_loss", 0) or 0
    if not final_take_profit:
        final_take_profit = strategy.get("take_profit", 0) or 0

    current = (data.get("mark_price", 0) or 0) if data else 0

    short_verdict = "维持原判" if "维持" in verdict_raw else ("推翻" if "推翻" in verdict_raw else verdict_raw)
    verdict_icon = "✅" if "维持" in short_verdict else "🔄"

    dir_icon = {"long": "🟢", "short": "🔴", "neutral": "⚪"}.get(final_direction, "⚪")
    dir_text = {"long": "做多", "short": "做空", "neutral": "观望"}.get(final_direction, "观望")
    size_text = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}.get(final_pos_size, "?")
    conf = strategy.get("confidence", "medium")
    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "?")

    header = f"### 📋 交易委员会 · 最终裁决 | {symbol} | {verdict_icon} {short_verdict} | {now}"
    decision_line = f"{dir_icon} {dir_text} | {size_text} | 置信度{conf_icon}"
    price_block = (
        f"现价 {current:.0f}  |  "
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
    return body


# 兼容旧名
def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> str:
    return format_final_decision(symbol, strategy, judge_result, data)
