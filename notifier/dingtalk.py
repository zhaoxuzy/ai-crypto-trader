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
DINGTALK_MAX_LENGTH = 18000  # 留一点余量，避免编码差异
SPLIT_MARKER = "\n\n--- (续) {part}/{total} ---\n\n"

def send_dingtalk_message(content: str, title: str = "策略推送") -> bool:
    if not content or not content.strip():
        logger.error(f"钉钉推送内容为空，已跳过: title={title}")
        return False

    webhook = os.getenv("DINGTALK_WEBHOOK_URL", "")
    secret = os.getenv("DINGTALK_SECRET", "")
    if not webhook:
        logger.error("未配置钉钉 Webhook")
        return False

    # 完整内容写入日志，方便回溯
    logger.info(f"完整消息内容 ({title}):\n{content}")

    # 超长内容自动分段
    if len(content) <= DINGTALK_MAX_LENGTH:
        return _send_single(webhook, secret, content, title)

    # 按段落粗略分割（尽量不在代码块中间切断）
    paragraphs = re.split(r'(\n\n)', content)
    segments = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > DINGTALK_MAX_LENGTH and current:
            segments.append(current)
            current = p
        else:
            current += p
    if current:
        segments.append(current)

    total = len(segments)
    success = True
    for i, seg in enumerate(segments, 1):
        seg_title = f"{title} ({i}/{total})"
        seg_content = seg
        if i > 1:
            seg_content = SPLIT_MARKER.format(part=i, total=total) + seg
        if not _send_single(webhook, secret, seg_content, seg_title):
            success = False
            logger.error(f"分段 {i}/{total} 发送失败")
    return success


def _send_single(webhook: str, secret: str, content: str, title: str) -> bool:
    """原有的单条发送逻辑"""
    ts = str(round(time.time() * 1000))
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
            logger.info(f"钉钉推送成功: {title}")
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


# ===================== 裁决文本解析（后备逻辑） =====================
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

    exec_start = re.search(r'🎯\s*合约策略', text)
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
    conf_text = {"high": "高", "medium": "中", "low": "低"}.get(conf, "?")

    header = f"### 策略｜{symbol} 🧠 建议 {dir_icon}{dir_text} {now}"
    first_line = f"现价 {current:.0f} | 入场 {entry_low:.0f}-{entry_high:.0f} | 止损 {stop_loss:.0f} | 止盈 {take_profit:.0f} | 置信度 {conf_text} | 仓位 {size_text}"

    full_reasoning = strategy.get("reasoning", "")
    if not full_reasoning:
        full_reasoning = strategy.get("full_reasoning") or strategy.get("raw") or "无推演内容"
    if isinstance(full_reasoning, str) and "\n" not in full_reasoning and len(full_reasoning) > 120:
        logger.warning(f"策略推演内容疑似单行大段文本，建议检查 reasoning 字段是否包含换行: {symbol}")

    body = (
        f"{header}\n"
        f"{first_line}\n\n"
        f"**推演过程**\n"
        f"```\n{_safe_code_block(full_reasoning)}\n```"
    )
    return body


def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    high = medium = low = 0
    if "severity_counts" in reviewer_report:
        sev = reviewer_report["severity_counts"]
        high = sev.get("高", sev.get("high", 0))
        medium = sev.get("中", sev.get("medium", 0))
        low = sev.get("低", sev.get("low", 0))
    elif "high" in reviewer_report or "严重" in reviewer_report:
        high = reviewer_report.get("high", reviewer_report.get("严重", 0))
        medium = reviewer_report.get("medium", reviewer_report.get("中等", 0))
        low = reviewer_report.get("low", reviewer_report.get("轻微", 0))
    else:
        report_text = reviewer_report.get("full_report", "")
        if report_text:
            high = report_text.count("严重")
            medium = report_text.count("中等")
            low = report_text.count("轻微")

    conclusion_icon = "⛔" if high > 0 else ("⚠️" if medium > 0 or low > 0 else "✅")
    conclusion_text = "驳回" if high > 0 else ("存疑" if medium > 0 or low > 0 else "通过")

    header = f"### 策略｜{symbol} ⚡ 审计 {conclusion_icon}{conclusion_text} {now}"
    first_line = f"严重 {high} | 中等 {medium} | 轻微 {low}"
    report = reviewer_report.get("full_report", "无审查报告")

    body = (
        f"{header}\n"
        f"{first_line}\n\n"
        f"**审计报告**\n"
        f"```\n{_safe_code_block(report)}\n```"
    )
    return body


def format_final_decision(symbol: str, strategy: dict, judge_result: dict = None, data: dict = None) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    # ---------- 优先从结构化字段提取 ----------
    if isinstance(judge_result, dict) and "judge_C" in judge_result:
        jc = judge_result["judge_C"]
        verdict = jc.get("final_verdict", "")          # "维持原判" / "推翻"
        final_direction = jc.get("final_direction", "neutral")
        final_pos_size = jc.get("final_position_size", "none")
        final_entry_low = jc.get("entry_price_low", 0) or 0
        final_entry_high = jc.get("entry_price_high", 0) or 0
        final_stop_loss = jc.get("stop_loss", 0) or 0
        final_take_profit = jc.get("take_profit", 0) or 0
        judge_full_text = jc.get("reasoning", "")
    else:
        # 回退到旧逻辑：从 judge_result 中提取文本再解析
        judge_full_text = ""
        if isinstance(judge_result, dict):
            judge_full_text = judge_result.get("content") or judge_result.get("full_report") or ""
        if not judge_full_text:
            judge_full_text = strategy.get("_judge_reasoning", "")

        parsed = _parse_judge_execution(judge_full_text) if judge_full_text else {}
        verdict = parsed.get("verdict", "维持原判")
        final_direction = parsed.get("direction", strategy.get("direction", "neutral"))
        final_pos_size = parsed.get("position_size", strategy.get("position_size", "none"))
        final_entry_low = parsed.get("entry_low", strategy.get("entry_price_low", 0) or 0)
        final_entry_high = parsed.get("entry_high", strategy.get("entry_price_high", 0) or 0)
        final_stop_loss = parsed.get("stop_loss", strategy.get("stop_loss", 0) or 0)
        final_take_profit = parsed.get("take_profit", strategy.get("take_profit", 0) or 0)

    # ---------- 标题 ----------
    if verdict == "维持原判":
        verdict_icon = "✅"
    else:
        verdict_icon = "🔄"

    header = f"### 策略｜{symbol} 📋 裁决 {verdict_icon}{verdict} {now}"

    # ---------- 方向/仓位/置信度 ----------
    dir_icon = {"long": "🟢", "short": "🔴", "neutral": "⚪"}.get(final_direction, "⚪")
    dir_text = {"long": "做多", "short": "做空", "neutral": "观望"}.get(final_direction, "观望")
    size_text = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}.get(final_pos_size, "?")
    conf = strategy.get("confidence", "medium")
    conf_text = {"high": "高", "medium": "中", "low": "低"}.get(conf, "?")

    current = (data.get("mark_price", 0) or 0) if data else 0

    # ---------- 首行：根据维持/推翻显示不同内容 ----------
    if verdict == "维持原判":
        first_line = (
            f"{dir_icon}{dir_text} | "
            f"现价 {current:.0f} | "
            f"入场 {final_entry_low:.0f}-{final_entry_high:.0f} | "
            f"止损 {final_stop_loss:.0f} | "
            f"止盈 {final_take_profit:.0f} | "
            f"置信度 {conf_text} | "
            f"仓位 {size_text}"
        )
    else:
        # 推翻时根据最终方向是否观望决定是否显示价格字段
        if final_direction == "neutral":
            first_line = (
                f"{dir_icon}{dir_text} | "
                f"现价 {current:.0f} | "
                f"置信度 {conf_text} | "
                f"仓位 {size_text}"
            )
        else:
            first_line = (
                f"{dir_icon}{dir_text} | "
                f"现价 {current:.0f} | "
                f"入场 {final_entry_low:.0f}-{final_entry_high:.0f} | "
                f"止损 {final_stop_loss:.0f} | "
                f"止盈 {final_take_profit:.0f} | "
                f"置信度 {conf_text} | "
                f"仓位 {size_text}"
            )

    if not judge_full_text:
        judge_full_text = "无裁决内容"

    body = (
        f"{header}\n"
        f"{first_line}\n\n"
        f"**裁决内容**\n"
        f"```\n{_safe_code_block(judge_full_text)}\n```"
    )
    return body


# 兼容旧名
def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> str:
    return format_final_decision(symbol, strategy, judge_result, data)
