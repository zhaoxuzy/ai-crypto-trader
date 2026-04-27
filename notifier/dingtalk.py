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
    """首席交易员推理文本格式化（仅用于初步信号）"""
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
        # 1. 构建简洁的标题
        if direction == "neutral":
            status_text = "⚠️审查推翻" if verdict == "推翻改为观望" else "⚠️风控驳回"
            title = f"策略信号：{symbol}｜📋 交易委员会：⚪ 观望 · {now} · {status_text}"
        else:
            emoji = "🟢" if direction == "long" else "🔴"
            text = "做多" if direction == "long" else "做空"
            size = strategy.get("position_size", "none")
            size_cn = {"light": "轻仓", "medium": "中仓", "heavy": "重仓"}.get(size, "")
            conf = strategy.get("confidence", "medium")
            conf_cn = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}.get(conf, "🟡中")
            status_tag = ""
            if verdict == "维持原判":
                status_tag = " · ✅审查确认"
            elif verdict == "修正参数":
                status_tag = " · 🔧审查修正"
            elif verdict == "降级执行":
                status_tag = " · ⚠️降级执行"
            elif verdict == "推翻改为反向操作":
                status_tag = " · 🔄推翻"
            title = f"策略信号：{symbol}｜📋 交易委员会：{emoji} {text} · {size_cn} · {conf_cn} · {now}{status_tag}"

        # 2. 直接取用 AI 提供的裁决理由，但只保留“审计指控”开始的部分
        reasoning_text = strategy.get("_judge_reasoning", "")
        if "审计指控" in reasoning_text:
            reasoning_text = reasoning_text[reasoning_text.find("审计指控"):]
        # 清理可能残存的风控报告片段
        if "风控审计官 - 审计报告" in reasoning_text:
            reasoning_text = reasoning_text.split("风控审计官 - 审计报告")[0].strip()
        reasoning_text = reasoning_text.strip()
        reasoning_block = f"📋 裁决理由：\n{reasoning_text}" if reasoning_text else "📋 裁决理由：\n无"

        # 3. 风险说明（清理残留的重复标签和说明）
        risk_note = strategy.get("risk_note", "请严格设置止损")
        risk_note = risk_note.replace("⚠️ 风险说明", "").strip()
        # 移除末尾可能重复出现的决议或观望字样
        risk_note = re.sub(r'\n*交易委员会决议.*', '', risk_note).strip()
        risk_note = re.sub(r'\n*⚠️\s*风险说明\s*观望.*', '', risk_note).strip()
        if verdict in ["推翻改为观望", "推翻改为反向操作"]:
            risk_note += f"\n\n交易委员会决议: {verdict}"
        risk_block = f"⚠️ 风险说明\n{risk_note}"

        # 4. 拼接最终消息
        parts = [title, "", reasoning_block, "", risk_block]
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
    report = report.replace('【风控审计官 - 审计报告】', '').strip()
    report = re.sub(r'#+\s*风控审计官.*\n', '', report).strip()
    return f"{title}\n\n{summary}\n\n📋 风控审计官 - 审计报告\n> {report}"


def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> str:
    """格式化交易委员会的最终裁决，直接复用最终信号的格式"""
    return format_strategy_message(symbol, strategy, data)