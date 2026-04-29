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

# 单个代码块最大行数（经验值，钉钉通常在20行左右开始折叠，我们设为10行更安全）
MAX_CODE_LINES = 10

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


def split_code_block(text: str, max_lines: int = MAX_CODE_LINES) -> list:
    """将大段文本拆分成多个小代码块，每个不超过指定行数"""
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    for line in lines:
        current_chunk.append(line)
        if len(current_chunk) >= max_lines:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return chunks


def format_strategy_message(symbol: str, strategy: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    direction = strategy.get("direction", "neutral")
    pos_size = strategy.get("position_size", "none")
    conf = strategy.get("confidence", "medium")
    entry_low = strategy.get("entry_price_low", 0)
    entry_high = strategy.get("entry_price_high", 0)
    stop_loss = strategy.get("stop_loss", 0)
    take_profit = strategy.get("take_profit", 0)
    current = data.get("mark_price", 0)

    dir_map = {"long": "🟢 做多", "short": "🔴 做空", "neutral": "⚪ 观望"}
    size_map = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}
    conf_map = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}

    line_title = f"**{symbol} 策略｜🧠首席交易员 · 🔍提交审计**  "
    line_decision = f"{dir_map.get(direction, '⚪ 观望')} · {size_map.get(pos_size, '未知')} · {conf_map.get(conf, '？')}   {now}  "
    line_price = (
        f"现价：{current:.0f}\n"
        f"入场：{entry_low:.0f}-{entry_high:.0f}\n"
        f"止损：{stop_loss:.0f}\n"
        f"止盈：{take_profit:.0f}  "
    )

    reasoning = strategy.get("reasoning", "无推演过程")
    
    # 最终策略优先从 final_strategy 字段提取，否则从 reasoning 里兜底
    final_strategy = strategy.get("final_strategy", "")
    if not final_strategy:
        match = re.search(r'最终合约策略[：:]\s*(.*?)(?=主动证伪|微观盘口确认|\n\s*\n|$)', reasoning, re.DOTALL)
        if match:
            final_strategy = match.group(1).strip()

    # 构建输出内容
    body_parts = [line_title, line_decision, line_price]

    # 如果有最终策略，用引用格式展示
    if final_strategy:
        strategy_text = "> " + final_strategy.replace('\n', '\n> ')
        body_parts.append(f"**最终策略**\n{strategy_text}")

    # 推演过程：按步骤拆分成多个小代码块，每个不超过 MAX_CODE_LINES 行
    steps = re.split(r'(?=第[一二三四五六七]步)', reasoning)
    if len(steps) > 1:
        body_parts.append("**推演过程**")
        for step in steps:
            step = step.strip()
            if not step:
                continue
            # 提取步骤标题
            m = re.match(r'(第[一二三四五六七]步[^：:]*[：:])', step)
            step_title = m.group(1) if m else ""
            step_content = step[len(step_title):].strip() if step_title else step
            
            # 将步骤内容拆分成多个小代码块
            chunks = split_code_block(step_content, MAX_CODE_LINES)
            for i, chunk in enumerate(chunks):
                if i == 0 and step_title:
                    body_parts.append(f"**{step_title}**\n```\n{chunk}\n```")
                else:
                    body_parts.append(f"```\n{chunk}\n```")
    else:
        # 无法按步骤分割，直接分块
        chunks = split_code_block(reasoning, MAX_CODE_LINES)
        body_parts.append("**推演过程**")
        for chunk in chunks:
            body_parts.append(f"```\n{chunk}\n```")

    return '\n'.join(body_parts)


def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    severity = reviewer_report.get("severity_counts", {"高": 0, "中": 0, "低": 0})
    high = severity.get("高", 0)
    medium = severity.get("中", 0)
    low = severity.get("低", 0)

    if high > 0:
        conclusion = "驳回"
    elif medium > 0 or low > 0:
        conclusion = "存疑"
    else:
        conclusion = "通过"

    line_title = f"**{symbol} 策略｜⚡风控审计官 · 📋审计完成**  "
    line_conclusion = f"结论：{conclusion}   {now}  "
    line_severity = f"🔴严重 {high}  🟡中等 {medium}  ⚪轻微 {low}  "

    report = reviewer_report.get("full_report", "无审查报告")
    # 审计报告也进行分块
    chunks = split_code_block(report, MAX_CODE_LINES)
    body_parts = [line_title, line_conclusion, line_severity, "**审计报告**"]
    for chunk in chunks:
        body_parts.append(f"```\n{chunk}\n```")

    return '\n'.join(body_parts)


def format_final_decision(symbol: str, strategy: dict, judge_result: dict = None, data: dict = None) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    direction = strategy.get("direction", "neutral")
    pos_size = strategy.get("position_size", "none")
    conf = strategy.get("confidence", "medium")
    verdict = strategy.get("_review_verdict", "")
    entry_low = strategy.get("entry_price_low", 0)
    entry_high = strategy.get("entry_price_high", 0)
    stop_loss = strategy.get("stop_loss", 0)
    take_profit = strategy.get("take_profit", 0)
    current = data.get("mark_price", 0) if data else 0

    dir_map = {"long": "🟢做多", "short": "🔴做空", "neutral": "⚪观望"}
    size_map = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}
    conf_map = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}

    verdict_map = {
        "维持原判": "✅维持原判",
        "修正参数": "🔧修正参数",
        "降级执行": "⚠️降级执行",
        "推翻改为观望": "🔄推翻→观望",
        "推翻改为反向操作": "🔄推翻"
    }

    line_title = f"**{symbol} 策略｜📋交易委员会 · ⚖️最终裁决**  "
    line_decision = f"{verdict_map.get(verdict, verdict)} · {dir_map.get(direction, '')} · {size_map.get(pos_size, '')} · {conf_map.get(conf, '')}   {now}  "
    line_price = (
        f"现价：{current:.1f}\n"
        f"入场：{entry_low:.0f}-{entry_high:.0f}\n"
        f"止损：{stop_loss:.0f}\n"
        f"止盈：{take_profit:.0f}  "
    )

    judge_content = strategy.get("_judge_reasoning", "")
    if not judge_content:
        judge_content = strategy.get("_judge_data", {}).get("reasoning", "")

    body_parts = [line_title, line_decision, line_price]
    if judge_content:
        body_parts.append("**裁决内容**")
        # 按“一、”“二、”等段落拆分裁决内容
        sections = re.split(r'(?=^[一二三四五六七八九十]、)', judge_content, flags=re.MULTILINE)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            chunks = split_code_block(section, MAX_CODE_LINES)
            for chunk in chunks:
                body_parts.append(f"```\n{chunk}\n```")

    return '\n'.join(body_parts)


# 兼容旧版调用
def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> str:
    return format_final_decision(symbol, strategy, judge_result, data)
