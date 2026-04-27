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

    reviewed = strategy.get("_reviewed", False)
    verdict = strategy.get("_review_verdict", "")
    preliminary = strategy.get("_preliminary", False)

    # ---------- 最终信号（审查后）----------
    if reviewed and not preliminary:
        # 风控审计驳回
        if direction == "neutral" and verdict == "推翻改为观望":
            title = f"策略信号：{symbol}｜⚡ 风控审计 · ⚠️驳回 · {now}"
            param = f"> 现价{data.get('mark_price', 0):.0f} · 入场0-0 · 止损0 · 止盈0 · 盈亏比N/A"
            summary_block = "📌 审计结论：原策略被推翻，改为观望"
            process_content = strategy.get("_reviewer_report", "")
            if process_content:
                process_block = f"📋 决议过程\n> \n> {process_content.strip()}"
            else:
                process_block = ""
            risk_raw = strategy.get("risk_note", "请严格设置止损")
            risk_lines = [f"> {line.strip()}" for line in risk_raw.split('\n') if line.strip()]
            if not risk_lines:
                risk_lines = ["> 请严格设置止损"]
            risk_block = "⚠️ 风险说明\n" + "\n".join(risk_lines)
            parts = [title, param, summary_block]
            if process_block:
                parts.append(process_block)
            parts.append(risk_block)
            return '\n\n'.join(parts)

        # 交易委员会
        emoji = "🟢" if direction == "long" else "🔴"
        text = "做多" if direction == "long" else "做空"
        size = strategy.get("position_size", "none")
        size_cn = {"light": "轻仓", "medium": "中仓", "heavy": "重仓"}.get(size, "")
        conf = strategy.get("confidence", "medium")
        conf_cn = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}.get(conf, "🟡中")
        title = f"策略信号：{symbol}｜📋 交易委员会：{emoji} {text} · {size_cn} · {conf_cn} · {now}"
        if verdict == "维持原判":
            title += " · ✅审查确认"
        elif verdict == "修正参数":
            title += " · 🔧审查修正"
        elif verdict == "降级执行":
            title += " · ⚠️降级执行"
        elif verdict == "推翻改为观望":
            title += " · 🔄推翻改为观望"

        entry_low = strategy.get("entry_price_low", 0)
        entry_high = strategy.get("entry_price_high", 0)
        stop = strategy.get("stop_loss", 0)
        tp = strategy.get("take_profit", 0)
        current = data.get("mark_price", 0)
        mid = (entry_low + entry_high) / 2 if entry_low and entry_high else 0
        risk = abs(mid - stop) if stop else 0
        reward = abs(tp - mid) if tp else 0
        rr = reward / risk if risk > 0 else 0
        rr_str = f"{rr:.2f}" if rr else "N/A"
        param = f"> 现价{current:.0f} · 入场{entry_low:.0f}-{entry_high:.0f} · 止损{stop:.0f} · 止盈{tp:.0f} · 盈亏比{rr_str}"

        summary_block = f"📌 决议：{verdict}"
        exec_plan = strategy.get("execution_plan", "")
        execution_block = f"🎯 执行指令\n> {exec_plan}" if exec_plan else ""

        # 决议过程：先审计报告，后法官裁决
        process_parts = []
        reviewer = strategy.get("_reviewer_report", "")
        if reviewer:
            process_parts.append(f"【风控审计官 - 审计报告】\n> {reviewer.strip()}")
        judge = strategy.get("_judge_reasoning", "")
        if judge:
            process_parts.append(f"【交易委员会裁决理由】\n> {judge.strip()}")
        process_block = "📋 决议过程\n> " + "\n\n> ".join(process_parts) if process_parts else ""

        risk_raw = strategy.get("risk_note", "请严格设置止损")
        risk_lines = [f"> {line.strip()}" for line in risk_raw.split('\n') if line.strip()]
        if not risk_lines:
            risk_lines = ["> 请严格设置止损"]
        risk_block = "⚠️ 风险说明\n" + "\n".join(risk_lines)

        parts = [title, param, summary_block]
        if execution_block:
            parts.append(execution_block)
        if process_block:
            parts.append(process_block)
        parts.append(risk_block)
        return '\n\n'.join(parts)

    # ---------- 初步信号（审查中）----------
    if direction == "neutral":
        title = f"策略信号：{symbol}｜🧠 首席交易员：⚪ 观望 · {now} · ⏳审查中"
        reasoning_raw = strategy.get("reasoning", "无推理过程")
        reasoning_block = f"### 🧠 交易员推理\n{format_reasoning(reasoning_raw)}"
        risk_raw = strategy.get("risk_note", "请严格设置止损")
        risk_lines = [f"> {line.strip()}" for line in risk_raw.split('\n') if line.strip()]
        if not risk_lines:
            risk_lines = ["> 请严格设置止损"]
        risk_block = "### ⚠️ 风险说明\n" + "\n".join(risk_lines)
        return f"{title}\n\n{reasoning_block}\n\n{risk_block}"

    emoji = "🟢" if direction == "long" else "🔴"
    text = "做多" if direction == "long" else "做空"
    size = strategy.get("position_size", "none")
    size_cn = {"light": "轻仓", "medium": "中仓", "heavy": "重仓"}.get(size, "")
    conf = strategy.get("confidence", "medium")
    conf_cn = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}.get(conf, "🟡中")
    title = f"策略信号：{symbol}｜🧠 首席交易员：{emoji} {text} · {size_cn} · {conf_cn} · {now} · ⏳审查中"

    entry_low = strategy.get("entry_price_low", 0)
    entry_high = strategy.get("entry_price_high", 0)
    stop = strategy.get("stop_loss", 0)
    tp = strategy.get("take_profit", 0)
    current = data.get("mark_price", 0)
    mid = (entry_low + entry_high) / 2 if entry_low and entry_high else 0
    risk = abs(mid - stop) if stop else 0
    reward = abs(tp - mid) if tp else 0
    rr = reward / risk if risk > 0 else 0
    rr_str = f"{rr:.2f}" if rr else "N/A"
    param = f"> 现价{current:.0f} · 入场{entry_low:.0f}-{entry_high:.0f} · 止损{stop:.0f} · 止盈{tp:.0f} · 盈亏比{rr_str}"

    reasoning_raw = strategy.get("reasoning", "无推理过程")
    reasoning_block = f"### 🧠 交易员推理\n{format_reasoning(reasoning_raw)}"

    risk_raw = strategy.get("risk_note", "请严格设置止损")
    risk_lines = [f"> {line.strip()}" for line in risk_raw.split('\n') if line.strip()]
    if not risk_lines:
        risk_lines = ["> 请严格设置止损"]
    risk_block = "### ⚠️ 风险说明\n" + "\n".join(risk_lines)

    return f"{title}\n\n{param}\n\n{reasoning_block}\n\n{risk_block}"


def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> str:
    """格式化风控审计官的审查报告"""
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")
    title = f"策略信号：{symbol}｜⚡ 风控审计 · {now}"
    report = reviewer_report.get("full_report", "无审查报告")
    severity = reviewer_report.get("severity_counts", {})
    summary = f"🔍 审计发现：严重问题{severity.get('高', 0)}个，中等问题{severity.get('中', 0)}个，轻微问题{severity.get('低', 0)}个"
    return f"{title}\n\n{summary}\n\n📋 风控审计官 - 审计报告\n> {report.strip()}"


def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> str:
    """格式化交易委员会的最终裁决，只展示关键决策，消除重复和格式标签"""
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")
    
    direction = strategy.get("direction", "neutral")
    verdict = strategy.get("_review_verdict", "维持原判")
    reasoning = strategy.get("_judge_reasoning", "")
    
    # ========== 标题行 ==========
    if direction == "neutral":
        title = f"策略信号：{symbol}｜📋 交易委员会：⚪ 观望 · {now} · ⚠️审查推翻"
    else:
        emoji = "🟢" if direction == "long" else "🔴"
        text = "做多" if direction == "long" else "做空"
        size = strategy.get("position_size", "none")
        size_cn = {"light": "轻仓", "medium": "中仓", "heavy": "重仓"}.get(size, "")
        conf = strategy.get("confidence", "medium")
        conf_cn = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}.get(conf, "🟡中")
        title = f"策略信号：{symbol}｜📋 交易委员会：{emoji} {text} · {size_cn} · {conf_cn} · {now}"
        if verdict == "维持原判":
            title += " · ✅审查确认"
        elif verdict == "修正参数":
            title += " · 🔧审查修正"
        elif verdict == "降级执行":
            title += " · ⚠️降级执行"
        elif verdict == "推翻改为反向操作":
            title += " · 🔄推翻"

    # ========== 参数卡片 ==========
    entry_low = strategy.get("entry_price_low", 0)
    entry_high = strategy.get("entry_price_high", 0)
    stop = strategy.get("stop_loss", 0)
    tp = strategy.get("take_profit", 0)
    current = data.get("mark_price", 0)
    param = f"> 现价{current:.0f} · 入场{entry_low:.0f}-{entry_high:.0f} · 止损{stop:.0f} · 止盈{tp:.0f}"

    # ========== 裁决摘要 ==========
    summary = f"📌 最终裁决：{verdict}"

    # ========== 执行指令（来自 strategy，已被 C 覆盖）==========
    exec_plan = strategy.get("execution_plan", "")
    execution = f"🎯 执行指令\n> {exec_plan}" if exec_plan else ""

    # ========== 裁决理由：深度清理，只保留审计指控的裁决内容 ==========
    clean_reasoning = reasoning
    
    # 1. 移除可能混入的风控审计报告原文
    if "风控审计官 - 审计报告" in clean_reasoning:
        clean_reasoning = clean_reasoning.split("风控审计官 - 审计报告")[0].strip()
    
    # 2. 移除 AI 输出中自带的📌、🎯等格式标签行以及策略参数行（方向/仓位/入场/止损/止盈/说明）
    clean_reasoning = re.sub(r'📌\s*最终判决[：:].*?(\n|$)', '', clean_reasoning)
    clean_reasoning = re.sub(r'🎯\s*执行指令[：:].*?(\n|$)', '', clean_reasoning)
    clean_reasoning = re.sub(r'\n\s*方向[：:].*?(\n|$)', '\n', clean_reasoning)
    clean_reasoning = re.sub(r'\n\s*仓位[：:].*?(\n|$)', '\n', clean_reasoning)
    clean_reasoning = re.sub(r'\n\s*入场区间[：:].*?(\n|$)', '\n', clean_reasoning)
    clean_reasoning = re.sub(r'\n\s*止损[：:].*?(\n|$)', '\n', clean_reasoning)
    clean_reasoning = re.sub(r'\n\s*止盈[：:].*?(\n|$)', '\n', clean_reasoning)
    clean_reasoning = re.sub(r'\n\s*说明[：:].*?(\n|$)', '\n', clean_reasoning)
    
    # 3. 清理多余空行
    clean_reasoning = re.sub(r'\n{3,}', '\n\n', clean_reasoning)
    clean_reasoning = clean_reasoning.strip()
    
    # 4. 如果清理后内容仍然很长，截取核心部分
    if len(clean_reasoning) > 500:
        clean_reasoning = clean_reasoning[:500] + "..."

    reasoning_block = f"📋 裁决理由\n> {clean_reasoning}" if clean_reasoning else ""

    # ========== 风险说明（来自 strategy，已被 C 覆盖）==========
    risk_note = strategy.get("risk_note", "请严格设置止损")
    risk_block = f"⚠️ 风险说明\n> {risk_note}"

    # ========== 拼接最终消息 ==========
    parts = [title, param, summary]
    if execution:
        parts.append(execution)
    if reasoning_block:
        parts.append(reasoning_block)
    parts.append(risk_block)
    return '\n\n'.join(parts)