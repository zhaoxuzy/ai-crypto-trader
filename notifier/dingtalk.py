import os
import requests
import re
import time
import hmac
import hashlib
import base64
import urllib.parse
from datetime import datetime, timezone, timedelta
from utils.logger import logger

# 北京时区 (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))


def send_dingtalk_message(msg: str, title: str = ""):
    """发送钉钉消息 (支持加签)"""
    webhook_url = os.getenv("DINGTALK_WEBHOOK_URL")
    secret = os.getenv("DINGTALK_SECRET")
    if not webhook_url:
        logger.warning("未配置 DINGTALK_WEBHOOK_URL，消息无法发送")
        return

    if secret:
        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{secret}"
        hmac_code = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        webhook_url = f"{webhook_url}&timestamp={timestamp}&sign={sign}" if "?" in webhook_url else f"{webhook_url}?timestamp={timestamp}&sign={sign}"

    data = {
        "msgtype": "markdown",
        "markdown": {
            "title": title.replace("\n", " ") if title else "策略推送",
            "text": msg
        }
    }

    try:
        resp = requests.post(webhook_url, json=data, timeout=10)
        if resp.status_code == 200:
            logger.info(f"钉钉推送成功: {title}")
        else:
            logger.error(f"钉钉推送失败: {resp.status_code} - {resp.text}")
    except Exception as e:
        logger.error(f"钉钉推送异常: {e}")


def format_strategy_message(symbol: str, strategy: dict, data: dict = None) -> str:
    """
    首席交易员初步方案消息
    包含：方向/仓位/现价/入场/止损/止盈 + 完整推演 (已移除结论速览)
    """
    direction = strategy.get("direction", "neutral")
    direction_emoji = {"long": "🟢做多", "short": "🔴做空", "neutral": "⚪观望"}.get(direction, "⚪观望")
    confidence = strategy.get("confidence", "?")
    position_size = strategy.get("position_size", "?")
    entry_low = strategy.get("entry_price_low", 0)
    entry_high = strategy.get("entry_price_high", 0)
    stop_loss = strategy.get("stop_loss", 0)
    take_profit = strategy.get("take_profit", 0)
    mark_price = data.get("mark_price", 0) if data else 0
    risk_note = strategy.get("risk_note", "")
    execution_plan = strategy.get("execution_plan", "")
    risk_reward = strategy.get("risk_reward_ratio", 0)
    reasoning = strategy.get("reasoning", "")

    msg = f"""### 策略｜{symbol} 初步方案 🧠 {datetime.now(BEIJING_TZ).strftime('%m-%d %H:%M')}

{direction_emoji} | 现价 {mark_price:.2f} | 入场 {entry_low:.2f}-{entry_high:.2f} | 止损 {stop_loss:.2f} | 止盈 {take_profit:.2f}
置信度 {'⭐'*3 if confidence=='high' else '⭐⭐' if confidence=='medium' else '⭐'} | 仓位 {'💰💰💰重仓' if position_size=='heavy' else '💰💰中仓' if position_size=='medium' else '💰轻仓' if position_size=='light' else '🚫无'}

---
### 📝 完整推演
{reasoning if reasoning else '（无推演内容）'}

"""
    if risk_note:
        msg += f"\n---\n### ❗️ 风险提示\n{risk_note}\n"
    if execution_plan:
        msg += f"\n---\n### ⚡ 执行\n{execution_plan}\n"
    if risk_reward > 0:
        msg += f"\n**盈亏比**：{risk_reward:.2f}"

    return msg


def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict = None) -> str:
    """
    审计官审查报告消息
    包含：核查清单 + 核心发现 + 结论
    """
    verdict = reviewer_report.get("overall_verdict", "未知")
    max_severity = reviewer_report.get("max_severity", "无")
    severity_summary = reviewer_report.get("severity_summary", {})
    full_report = reviewer_report.get("full_report", "")

    # 构建核查清单摘要
    checklist_summary = ""
    step_audits = reviewer_report.get("step_audits", [])
    if step_audits:
        for audit in step_audits:
            step_num = audit.get("step", "?")
            v = audit.get("verdict", "未知")
            emoji = "✅" if v == "合格" else "❌" if v == "严重错误" else "⚠️"
            checklist_summary += f"{emoji} 步骤{step_num}: {v}\n"

    sev_text = f"🔴严重 {severity_summary.get('严重',0)} | 🟡中等 {severity_summary.get('中等',0)} | 🟢轻度 {severity_summary.get('轻度',0)}"
    direction = strategy.get("direction", "?")
    position_size = strategy.get("position_size", "?")

    msg = f"""### 策略｜{symbol} 审计报告 📋 {datetime.now(BEIJING_TZ).strftime('%m-%d %H:%M')}

{verdict} | 最高严重性：{max_severity} | {sev_text}
原方向：{direction} | 原仓位：{position_size}

---
{full_report}

---
### ⚡ 核查清单
{checklist_summary if checklist_summary else '无步骤审查数据'}
"""
    return msg


def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict = None) -> str:
    """
    交易委员会最终裁决消息
    包含：方向/仓位/置信度 + 裁决明细 + 合约执行单
    """
    verdict = judge_result.get("final_verdict", "未知")
    final_direction = judge_result.get("final_direction", "neutral")
    final_direction_emoji = "🟢做多" if final_direction == "long" else "🔴做空" if final_direction == "short" else "⚪观望"
    final_confidence = judge_result.get("final_confidence", "?")
    final_position = judge_result.get("final_position_size", "?")
    entry_low = judge_result.get("entry_price_low", 0)
    entry_high = judge_result.get("entry_price_high", 0)
    stop_loss = judge_result.get("stop_loss", 0)
    take_profit = judge_result.get("take_profit", 0)
    execution_plan = judge_result.get("execution_plan", "")
    risk_note = judge_result.get("risk_note", "")
    final_reasoning = judge_result.get("final_reasoning", "")
    audit_responses = judge_result.get("audit_responses", [])
    weighted_signal = judge_result.get("weighted_signal", {})
    mark_price = data.get("mark_price", 0) if data else 0

    # 构建裁决明细
    response_text = ""
    if audit_responses:
        for resp in audit_responses:
            step = resp.get("step", "?")
            issue = resp.get("issue", "")
            adopted = "成立 ✅" if resp.get("adopted") else "不成立 ❌"
            reason = resp.get("reason", "")
            response_text += f"**指控{step}**：{issue}\n→ 裁决：{adopted}\n→ 理由：{reason}\n\n"

    # 构建执行单
    msg = f"""### 策略｜{symbol} 最终裁决 ⚖️ {datetime.now(BEIJING_TZ).strftime('%m-%d %H:%M')}

{verdict} | {final_direction_emoji} | 置信度 {'⭐'*3 if final_confidence=='high' else '⭐⭐' if final_confidence=='medium' else '⭐'} | 仓位 {'💰💰💰重仓' if final_position=='heavy' else '💰💰中仓' if final_position=='medium' else '💰轻仓' if final_position=='light' else '🚫无'}

---
### 📌 裁决理由
{final_reasoning if final_reasoning else '无'}

"""
    if response_text:
        msg += f"---\n### 📋 裁决明细\n{response_text}"

    msg += f"""---
### 🎯 合约执行单
现价：{mark_price:.2f}
入场：{entry_low:.2f} - {entry_high:.2f}
止损：{stop_loss:.2f}
止盈：{take_profit:.2f}
盈亏比：{((take_profit - mark_price) / (mark_price - stop_loss)) if (mark_price - stop_loss) > 0 else 0:.2f}:1
执行：{execution_plan if execution_plan else '无'}
风险：{risk_note if risk_note else '无'}
"""
    if weighted_signal:
        weights_str = " | ".join([f"{k}: {v}" for k, v in weighted_signal.items()])
        msg += f"\n---\n### ⚖️ 加权信号\n{weights_str}\n"

    return msg


def format_final_decision(symbol: str, strategy: dict, judge_result: dict, data: dict = None) -> str:
    """最终策略推送消息（复用委员会裁决消息）"""
    return format_judge_message(symbol, strategy, judge_result, data)
