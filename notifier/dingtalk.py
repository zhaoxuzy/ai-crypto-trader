"""
notifier/dingtalk.py — 钉钉推送 (北京时间 + 分层折叠模板)
"""
import os
import json
import requests
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

# ---------- 格式化辅助 ----------
def _to_quote(text: str) -> str:
    """将多行文本转换为 Markdown 引用块（每行前加 > ）"""
    if not text:
        return "> （无内容）"
    lines = text.split('\n')
    quoted = []
    for line in lines:
        if line.strip() == "":
            quoted.append("> ")
        else:
            quoted.append(f"> {line}")
    return '\n'.join(quoted)

def _normalize_direction(raw: str) -> str:
    """强制将方向转为英文小写，兼容中文输入"""
    if not raw:
        return "neutral"
    raw = raw.strip().lower()
    if raw in ("long", "short", "neutral"):
        return raw
    mapping = {"做多": "long", "做空": "short", "观望": "neutral"}
    return mapping.get(raw, "neutral")

def _direction_emoji(direction: str) -> str:
    dir_en = _normalize_direction(direction)
    m = {"long": "🟢做多", "short": "🔴做空", "neutral": "⚪观望"}
    return m.get(dir_en, "⚪观望")

def _conf_stars(confidence: str) -> str:
    m = {"high": "⭐⭐⭐高", "medium": "⭐⭐中", "low": "⭐低"}
    return m.get(confidence.lower() if confidence else "", "⭐⭐中")

def _pos_emoji(position_size: str) -> str:
    size = (position_size or "").lower()
    m = {"heavy": "💰💰💰重仓", "medium": "💰💰中仓", "light": "💰轻仓", "none": "🚫无"}
    return m.get(size, "🚫无")

def _audit_emoji(verdict: str) -> str:
    m = {"通过": "🟢通过", "存疑": "🟡存疑", "驳回": "🔴驳回"}
    return m.get(verdict, f"⚪{verdict}")

def _smart_truncate(text: str, max_len: int, head_ratio: float = 0.75) -> str:
    """智能截断：保留头部与尾部，中间省略。"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    head_len = int(max_len * head_ratio)
    tail_len = max_len - head_len - len("\n\n...（中间分析省略）...\n\n")
    if tail_len < 100:
        tail_len = min(100, max_len // 2)
    head = text[:head_len]
    tail = text[-tail_len:]
    return head + "\n\n...（中间分析省略）...\n\n" + tail

def _format_price(val, decimals=2):
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return "0.00"

# ---------- 消息构建 ----------
def format_strategy_message(symbol: str, strategy: dict, data: dict = None) -> str:
    """初步方案消息：方向+价位+仓位+推理折叠（无摘要）"""
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
    line1 = f"{_direction_emoji(direction)} | 现价 {_format_price(mark)} | "
    line1 += f"入场 {_format_price(entry_low)}-{_format_price(entry_high)}\n"
    line2 = f"止损 {_format_price(stop)} | 止盈 {_format_price(profit)} | "
    line2 += f"置信度 {_conf_stars(confidence)} | 仓位 {_pos_emoji(position)}\n\n"

    # 完整推演截断至2000字，保留头尾
    truncated = _smart_truncate(reasoning, 2000, head_ratio=0.6)
    msg = header + line1 + line2
    msg += "> **完整推演**\n"
    msg += _to_quote(truncated) if truncated else "> （无推演内容）"
    return msg

def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict = None) -> str:
    """审计报告消息：结论+严重性(来自report字段)+原策略+审计全文折叠"""
    verdict = reviewer_report.get("verdict", "未知")
    # 直接使用审计官返回的 severity_counts，若缺失则回退空
    severity = reviewer_report.get("severity_counts", {})
    if not isinstance(severity, dict):
        severity = {}
    severe = severity.get("严重", 0)
    medium = severity.get("中等", 0)
    low = severity.get("轻度", 0)
    # 兼容可能的英文键名(若将来有)
    if severe == 0 and "critical" in severity:
        severe = severity["critical"]
    if medium == 0 and "medium" in severity:
        medium = severity["medium"]
    if low == 0 and "low" in severity:
        low = severity["low"]

    full_report = reviewer_report.get("full_report", "")
    orig_direction = _normalize_direction(strategy.get("direction", "neutral"))
    orig_position = strategy.get("position_size", "无")

    header = f"### 策略｜{symbol} 审计报告 📋 {_beijing_time()}\n"
    line1 = f"{_audit_emoji(verdict)} | 严重：{severe}  中等：{medium}  轻微：{low}\n"
    line2 = f"原方向：{_direction_emoji(orig_direction)} | 原仓位：{_pos_emoji(orig_position)}\n\n"

    # 审计报告截断至1500字，保留头尾
    truncated_report = _smart_truncate(full_report, 1500, head_ratio=0.6)

    msg = header + line1 + line2
    msg += "> **审计报告**\n"
    msg += _to_quote(truncated_report) if truncated_report else "> （无审计内容）"
    return msg

def format_final_decision(symbol: str, strategy: dict, judge_result: dict, data: dict = None) -> str:
    """最终计划消息：维持/推翻+最终决策+原方向对比+裁决理由折叠"""
    verdict = judge_result.get("final_verdict", "维持原判")
    final_direction = _normalize_direction(judge_result.get("final_direction", strategy.get("direction", "neutral")))
    final_confidence = judge_result.get("final_confidence", strategy.get("confidence", "中"))
    final_position = judge_result.get("final_position_size", strategy.get("position_size", "无"))
    entry_low = judge_result.get("entry_price_low", 0) or 0
    entry_high = judge_result.get("entry_price_high", 0) or 0
    stop = judge_result.get("stop_loss", 0) or 0
    profit = judge_result.get("take_profit", 0) or 0
    final_reasoning = judge_result.get("final_reasoning", "")
    orig_direction = _normalize_direction(strategy.get("direction", "neutral"))
    mark = data.get('mark_price', 0) if data else 0

    # 裁决状态短标签
    verdict_short = {"维持原判": "✅维持", "推翻": "🔄推翻", "修改执行": "🔧修改执行"}.get(verdict, verdict)
    # 方向显示：当前方向 (原方向) —— 简洁表示
    dir_display = f"{_direction_emoji(final_direction)} (原{_direction_emoji(orig_direction)})"

    header = f"### 策略｜{symbol} 最终计划 ⚖️ {_beijing_time()}\n"
    line1 = f"{verdict_short} | {dir_display} | 现价 {_format_price(mark)}\n"
    line2 = f"入场 {_format_price(entry_low)}-{_format_price(entry_high)} | "
    line2 += f"止损 {_format_price(stop)} | 止盈 {_format_price(profit)}\n"
    line3 = f"置信度 {_conf_stars(final_confidence)} | 仓位 {_pos_emoji(final_position)}\n\n"

    # 裁决理由截断至2000字，优先保留对审计指控的回应
    truncated_reasoning = _smart_truncate(final_reasoning, 2000, head_ratio=0.65)

    msg = header + line1 + line2 + line3
    msg += "> **裁决理由**\n"
    msg += _to_quote(truncated_reasoning) if truncated_reasoning else "> （无裁决理由）"

    return msg

# 保留旧接口兼容性
format_judge_message = format_final_decision