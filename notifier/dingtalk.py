import os
import json
import requests
import re
from datetime import datetime, timezone, timedelta
from utils.logger import logger

BEIJING_TZ = timezone(timedelta(hours=8))

def _beijing_time():
    return datetime.now(BEIJING_TZ).strftime('%m-%d %H:%M')

def _send_dingtalk(webhook_url, payload):
    try:
        headers = {"Content-Type": "application/json"}
        resp = requests.post(webhook_url, data=json.dumps(payload), headers=headers, timeout=10)
        if resp.status_code == 200 and resp.json().get("errcode") == 0:
            return True
        else:
            logger.error(f"钉钉返回错误: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"钉钉推送异常: {e}")
        return False

def send_dingtalk_message(text: str, title: str = "策略通知") -> bool:
    webhook_url = os.getenv("DINGTALK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("未配置 DINGTALK_WEBHOOK_URL")
        return False
    payload = {"msgtype": "markdown", "markdown": {"title": title[:30], "text": text}}
    return _send_dingtalk(webhook_url, payload)

# ---------- 辅助 ----------
def _to_quote(text: str) -> str:
    if not text:
        return "> （无内容）"
    return '\n'.join(f"> {line}" if line.strip() else "> " for line in text.split('\n'))

def _normalize_direction(raw: str) -> str:
    if not raw: return "neutral"
    raw = raw.strip().lower()
    mapping = {"做多":"long","做空":"short","观望":"neutral"}
    return mapping.get(raw, raw)

def _direction_emoji(direction: str) -> str:
    dir_en = _normalize_direction(direction)
    m = {"long":"🟢做多","short":"🔴做空","neutral":"⚪观望"}
    return m.get(dir_en, "⚪观望")

def _conf_stars(confidence: str) -> str:
    m = {"high":"⭐⭐⭐高","medium":"⭐⭐中","low":"⭐低"}
    return m.get(confidence.lower() if confidence else "", "⭐⭐中")

def _pos_emoji(position_size: str) -> str:
    size = (position_size or "").lower()
    m = {"heavy":"💰💰💰重仓","medium":"💰💰中仓","light":"💰轻仓","none":"🚫无"}
    return m.get(size, "🚫无")

def _audit_verdict_emoji(verdict: str) -> str:
    m = {"通过":"🟢通过","存疑":"🟡存疑","驳回":"🔴驳回",
         "合格":"✅","存在瑕疵":"⚠️","严重错误":"❌"}
    return m.get(verdict, f"⚪{verdict}")

def _smart_truncate(text: str, max_len: int, head_ratio: float = 0.75) -> str:
    if not text or len(text) <= max_len:
        return text
    head_len = int(max_len * head_ratio)
    tail_len = max_len - head_len - len("\n\n...（中间省略）...\n\n")
    if tail_len < 100:
        tail_len = min(100, max_len // 2)
    return text[:head_len] + "\n\n...（中间省略）...\n\n" + text[-tail_len:]

def _format_price(val, decimals=2):
    try: return f"{float(val):.{decimals}f}"
    except: return "0.00"

def _extract_step_summary(reasoning: str) -> str:
    """提取七步标题作为摘要"""
    if not reasoning: return "> （无推演内容）"
    steps = []
    for line in reasoning.split('\n'):
        if re.match(r'【步骤\d】', line):
            steps.append(line.strip()[:80])
    if not steps:
        steps = [l.strip()[:80] for l in reasoning.split('\n') if l.strip()][:7]
    return '\n'.join(f"> {s}" for s in steps)

# ---------- 消息构建 ----------
def format_strategy_message(symbol: str, strategy: dict, data: dict = None) -> str:
    direction = _normalize_direction(strategy.get("direction", "neutral"))
    confidence = strategy.get("confidence", "中")
    position = strategy.get("position_size", "无")
    entry_low = strategy.get("entry_price_low", 0) or 0
    entry_high = strategy.get("entry_price_high", 0) or 0
    stop = strategy.get("stop_loss", 0) or 0
    profit = strategy.get("take_profit", 0) or 0
    reasoning = strategy.get("reasoning", "")
    mark = data.get('mark_price', 0) if data else 0

    header = f"### 策略｜{symbol} 初步方案 ⏳ {_beijing_time()}\n"
    line1 = f"{_direction_emoji(direction)} | 现价 {_format_price(mark)} | 入场 {_format_price(entry_low)}-{_format_price(entry_high)}\n"
    line2 = f"止损 {_format_price(stop)} | 止盈 {_format_price(profit)} | 置信度 {_conf_stars(confidence)} | 仓位 {_pos_emoji(position)}\n\n"

    # 七步摘要
    msg = header + line1 + line2
    msg += "> **七步推演概要**\n"
    msg += _extract_step_summary(reasoning) + "\n\n"
    # 完整推理截断
    truncated = _smart_truncate(reasoning, 1800, head_ratio=0.6)
    msg += "> **完整推演**\n" + _to_quote(truncated)
    return msg

def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict = None) -> str:
    overall = reviewer_report.get("overall_verdict","未知")
    severity = reviewer_report.get("severity_summary", {})
    step_audits = reviewer_report.get("step_audits", [])
    full_report = reviewer_report.get("full_report", "")
    orig_dir = _normalize_direction(strategy.get("direction","neutral"))
    orig_pos = strategy.get("position_size","无")

    header = f"### 策略｜{symbol} 审计报告 📋 {_beijing_time()}\n"
    line1 = f"{_audit_verdict_emoji(overall)} | 严重：{severity.get('严重',0)} 中等：{severity.get('中等',0)} 轻微：{severity.get('轻度',0)}\n"
    line2 = f"原方向：{_direction_emoji(orig_dir)} | 原仓位：{_pos_emoji(orig_pos)}\n\n"

    # 七步审计摘要
    audit_line = "> **七步审计**\n> "
    if step_audits:
        parts = []
        for sa in step_audits:
            st = sa.get("step","?")
            v = _audit_verdict_emoji(sa.get("verdict",""))
            parts.append(f"步骤{st}：{v}")
        audit_line += " | ".join(parts)
    else:
        audit_line += "（无分步审计数据）"
    audit_line += "\n\n"

    # 被忽略指标提示（若有）
    ignored_note = reviewer_report.get("ignored_indicators","")
    if ignored_note:
        audit_line += f"> **被忽略的关键指标**：{ignored_note}\n\n"

    # 报告正文
    truncated = _smart_truncate(full_report, 1500, head_ratio=0.6)
    msg = header + line1 + line2 + audit_line
    msg += "> **审计报告全文**\n" + _to_quote(truncated)
    return msg

def format_final_decision(symbol: str, strategy: dict, judge_result: dict, data: dict = None) -> str:
    verdict = judge_result.get("final_verdict","维持原判")
    final_dir = _normalize_direction(judge_result.get("final_direction", strategy.get("direction")))
    orig_dir = _normalize_direction(strategy.get("direction","neutral"))
    final_conf = judge_result.get("final_confidence", strategy.get("confidence","中"))
    final_pos = judge_result.get("final_position_size", strategy.get("position_size","无"))
    entry_low = judge_result.get("entry_price_low", 0) or 0
    entry_high = judge_result.get("entry_price_high", 0) or 0
    stop = judge_result.get("stop_loss", 0) or 0
    profit = judge_result.get("take_profit", 0) or 0
    reasoning = judge_result.get("final_reasoning","")
    mark = data.get('mark_price', 0) if data else 0

    verdict_label = {"维持原判":"✅维持","推翻":"🔄推翻","修改执行":"🔧修改执行"}.get(verdict, verdict)
    dir_display = f"{_direction_emoji(final_dir)} (原{_direction_emoji(orig_dir)})"

    header = f"### 策略｜{symbol} 最终计划 ⚖️ {_beijing_time()}\n"
    line1 = f"{verdict_label} | {dir_display} | 现价 {_format_price(mark)}\n"
    line2 = f"入场 {_format_price(entry_low)}-{_format_price(entry_high)} | 止损 {_format_price(stop)} | 止盈 {_format_price(profit)}\n"
    line3 = f"置信度 {_conf_stars(final_conf)} | 仓位 {_pos_emoji(final_pos)}\n\n"

    # 加权信号
    ws = judge_result.get("weighted_signal", {})
    if ws:
        signals = f"宏观：{ws.get('step2','-')} | 动能：{ws.get('step3','-')} | 博弈：{ws.get('step4','-')} | 预期差：{ws.get('step5','-')} | 跨币种：{ws.get('step6','-')} | 综合分：{ws.get('composite_score','-')}"
        line3 += f"> 加权信号：{signals}\n\n"

    # 对审计指控的回应
    responses = judge_result.get("audit_responses", [])
    if responses:
        line3 += "> **对审计指控的回应**\n"
        for res in responses[:5]:
            step = res.get("step","")
            desc = res.get("issue","")
            adopted = "采信" if res.get("adopted") else "驳回"
            reason = res.get("reason","")
            line3 += f"> ·【步骤{step}】{desc} → {adopted}（{reason}）\n"
        line3 += "\n"

    # 逆周期警告
    if ws and ws.get("step2") != "neutral" and _normalize_direction(final_dir) != "neutral":
        macro_dir = "long" if "bull" in ws.get("step2","").lower() else ("short" if "bear" in ws.get("step2","").lower() else "neutral")
        if macro_dir != final_dir:
            line3 += "> ⚠️ **逆周期交易警告**：策略方向与宏观底色不一致，请严格控制风险\n\n"

    # 裁决理由
    truncated = _smart_truncate(reasoning, 1800, head_ratio=0.65)
    msg = header + line1 + line2 + line3
    msg += "> **裁决理由**\n" + _to_quote(truncated)
    return msg

# 兼容旧版
format_judge_message = format_final_decision
