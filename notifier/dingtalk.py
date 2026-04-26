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


def format_reasoning(text: str) -> str:
    if not text:
        return "> 无推理过程"
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    titles = [
        "第[一二三四五六七八九]步[：:]",
        "分析数据[：:]",
        "第一反应[：:]",
        "自我质疑[：:]",
        "最终结论[：:]",
        "交叉验证与裁决[：:]",
        "价格路径推演[：:]",
        "推理自检[：:]",
        "入场区间[：:]",
        "止损位[：:]",
        "止盈位[：:]",
        "主动证伪信号[：:]",
        "微观盘口确认[：:]"
    ]
    for title in titles:
        text = re.sub(rf'(?<!\n)({title})', r'\n\1', text)
    lines = text.split('\n')
    quoted = []
    for line in lines:
        line = line.strip()
        if not line:
            quoted.append('> ')
            continue
        if re.match(r'^第[一二三四五六七八九]步[：:]', line):
            line = re.sub(r'^(第[一二三四五六七八九]步)', r'**\1**', line)
        elif re.match(r'^(分析数据|第一反应|自我质疑|最终结论|交叉验证与裁决|价格路径推演|推理自检|入场区间|止损位|止盈位|主动证伪信号|微观盘口确认)[：:]', line):
            line = re.sub(r'^([^：:]+)', r'**\1**', line)
        quoted.append(f'> {line}' if not line.startswith('>') else line)
    cleaned = []
    prev_empty = False
    for q in quoted:
        is_empty = (q.strip() == '>')
        if is_empty and prev_empty:
            continue
        cleaned.append(q)
        prev_empty = is_empty
    return '\n'.join(cleaned)


def format_strategy_message(symbol: str, strategy: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")
    direction = strategy.get("direction", "neutral")

    if direction == "neutral":
        title = f"## ⚪ 观望 {symbol} · 🔴低 · {now}"
        param = f"> 现价{data.get('mark_price', 0):.0f} · 入场0-0 · 止损0 · 止盈0"
    else:
        emoji = "🟢" if direction == "long" else "🔴"
        text = "做多" if direction == "long" else "做空"
        size = strategy.get("position_size", "none")
        size_cn = {"light": "轻仓", "medium": "中仓", "heavy": "重仓"}.get(size, "")
        conf = strategy.get("confidence", "medium")
        conf_cn = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}.get(conf, "🟡中")
        parts = [f"{emoji} {text} {symbol}"]
        if size_cn:
            parts.append(size_cn)
        parts.append(conf_cn)
        parts.append(now)
        title = "## " + " · ".join(parts)
        entry_low = strategy.get("entry_price_low", 0)
        entry_high = strategy.get("entry_price_high", 0)
        stop = strategy.get("stop_loss", 0)
        tp = strategy.get("take_profit", 0)
        current = data.get("mark_price", 0)
        param = f"> 现价{current:.0f} · 入场{entry_low:.0f}-{entry_high:.0f} · 止损{stop:.0f} · 止盈{tp:.0f}"

    reasoning_raw = strategy.get("reasoning", "无推理过程")
    reasoning_block = format_reasoning(reasoning_raw)

    risk_raw = strategy.get("risk_note", "请严格设置止损")
    risk_lines = [f"> {line.strip()}" for line in risk_raw.split('\n') if line.strip()]
    if not risk_lines:
        risk_lines = ["> 请严格设置止损"]
    risk_block = "> ### ⚠️ 风险说明\n" + "\n".join(risk_lines)

    atr = data.get("atr_15m", 0)
    funding = data.get("funding_rate", 0)
    oi_chg = data.get("oi_change_24h", 0)
    cvd = data.get("cvd_slope", 0)
    cvd_dir = "↗" if cvd > 0 else ("↘" if cvd < 0 else "→")
    fg = data.get("fear_greed", 50)
    foot = f"📎 ATR{atr:.0f} · 费率{funding:.4f}% · OI{oi_chg:+.1f}% · CVD{cvd_dir} · 贪婪{fg}"

    return f"{title}\n\n{param}\n\n### 🧠 交易员推理\n{reasoning_block}\n\n{risk_block}\n\n{foot}"


def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")
    
    verdict = reviewer_report.get("verdict", "通过")
    severity = reviewer_report.get("severity_counts", {})
    full_report = reviewer_report.get("full_report", "")
    
    title = f"## 🔍 审查报告 {symbol} · {now}"
    if verdict == "驳回":
        title += " · ⚠️驳回"
    elif verdict == "存疑":
        title += " · ⚡存疑"
    else:
        title += " · ✅通过"
    
    summary = f"> 审查结论：{verdict}\n"
    if severity:
        summary += f"> 严重性统计：高={severity.get('高', 0)} 中={severity.get('中', 0)} 低={severity.get('低', 0)}\n"
    
    report_block = ""
    section_titles = [
        "一、数据与解读错误",
        "二、逻辑错误", 
        "三、关键反证提示",
        "四、博弈层面审视"
    ]
    
    for line in full_report.split('\n'):
        line = line.strip()
        if not line:
            report_block += "> \n"
            continue
            
        is_title = False
        for title_text in section_titles:
            if line.startswith(title_text):
                report_block += f"> **{line}**\n"
                is_title = True
                break
        
        if not is_title:
            if line.startswith('>'):
                report_block += f"{line}\n"
            else:
                report_block += f"> {line}\n"
    
    return f"{title}\n\n{summary}\n\n{report_block}"


def format_judge_message(symbol: str, strategy: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    direction = strategy.get("direction", "neutral")
    verdict = strategy.get("_review_verdict", "维持原判")
    judge_reasoning = strategy.get("_judge_reasoning", "")
    exec_plan = strategy.get("execution_plan", "")
    original_direction = strategy.get("_original_direction", "")

    verdict_emoji_map = {
        "维持原判": "✅",
        "修正参数": "🔧",
        "降级执行": "⬇️",
        "推翻改为观望": "⚠️",
        "推翻改为反向操作": "🔄"
    }
    verdict_emoji = verdict_emoji_map.get(verdict, "•")

    # ----- 标题 -----
    if direction == "neutral":
        title = f"## ⚪ 最终裁决 {symbol} · 观望 · {now} · {verdict_emoji}{verdict}"
    else:
        emoji = "🟢" if direction == "long" else "🔴"
        text = "做多" if direction == "long" else "做空"
        size = strategy.get("position_size", "none")
        size_cn = {"light": "轻仓", "medium": "中仓", "heavy": "重仓"}.get(size, "")
        conf = strategy.get("confidence", "medium")
        conf_cn = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}.get(conf, "🟡中")
        parts = [f"{emoji} 最终裁决 {symbol}"]
        if size_cn:
            parts.append(size_cn)
        parts.append(conf_cn)
        parts.append(now)
        parts.append(f"{verdict_emoji}{verdict}")
        title = "## " + " · ".join(parts)

    # ----- 参数卡片 (移除盈亏比) -----
    entry_low = strategy.get("entry_price_low", 0)
    entry_high = strategy.get("entry_price_high", 0)
    stop = strategy.get("stop_loss", 0)
    tp = strategy.get("take_profit", 0)
    current = data.get("mark_price", 0)

    if direction == "neutral":
        param = f"> 现价{current:.0f} · 入场0-0 · 止损0 · 止盈0"
    else:
        param = f"> 现价{current:.0f} · 入场{entry_low:.0f}-{entry_high:.0f} · 止损{stop:.0f} · 止盈{tp:.0f}"

    # ----- 判决摘要 (增加反向操作描述) -----
    reversed_direction = False
    if original_direction and original_direction != direction:
        reversed_direction = True

    verdict_summary = {
        "维持原判": "C认为A的策略无明显瑕疵，予以维持。",
        "修正参数": "C认为A的方向正确，但对参数进行了优化修正。",
        "降级执行": "C认为A的逻辑有瑕疵，降仓后仍可执行。",
        "推翻改为观望": "C认为A的方向存在致命问题，改为观望。",
        "推翻改为反向操作": "C认为A的方向与市场结构完全相反，已转向反向操作。"
    }
    summary_text = verdict_summary.get(verdict, "")

    if reversed_direction and verdict not in verdict_summary:
        summary_text = f"C认为A的{original_direction}方向存在致命问题，改为完全相反的{direction}方向。"

    summary_block = f"> **📌 判决：{verdict}**\n> {summary_text}\n" if summary_text else ""

    # ----- 裁决理由 -----
    reasoning_block = ""
    if judge_reasoning:
        lines = judge_reasoning.split('\n')
        reasoning_block = "> ### 📋 裁决理由\n"
        for line in lines:
            stripped = line.strip()
            if not stripped:
                reasoning_block += "> \n"
                continue
            if re.match(r'^\s*[\d]+[\.、\)]|^\s*[-•]', stripped):
                reasoning_block += f"> {stripped}\n"
            else:
                reasoning_block += f"> {stripped}\n"

    # ----- 执行指令 -----
    if exec_plan and direction != "neutral":
        reasoning_block += f"\n> ### 🎯 执行指令\n> {exec_plan}\n"

    # ----- 风险说明 -----
    risk_raw = strategy.get("risk_note", "请严格设置止损")
    risk_lines = [f"> {line.strip()}" for line in risk_raw.split('\n') if line.strip()]
    if not risk_lines:
        risk_lines = ["> 请严格设置止损"]
    risk_block = "> ---\n> ### ⚠️ 风险说明\n" + "\n".join(risk_lines)

    return f"{title}\n\n{param}\n\n{summary_block}\n{reasoning_block}\n{risk_block}"
