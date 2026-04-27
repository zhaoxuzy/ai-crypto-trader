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
        # 1. 标题
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

        # 2. 执行指令：直接取自C修正后的版本
        exec_plan = strategy.get("execution_plan", "无具体指令")
        execution_block = f"🎯 执行指令\n{exec_plan}"

        # 3. 完整裁决内容：直接使用 AI 返回的原始文本，不进行任何修改
        reasoning_raw = strategy.get("_judge_reasoning", "")
        # 只做极简处理：如果文本为空，给一个占位符
        if not reasoning_raw.strip():
            reasoning_raw = "无裁决理由"

        # 4. 风险说明：直接使用 C 覆盖后的版本，不添加额外标题，让 AI 自行处理格式
        risk_raw = strategy.get("risk_note", "请严格设置止损")

        # 5. 拼接最终消息：标题 + 执行指令 + 原始AI文本 + 风险说明
        # 注意：不再添加任何额外的“📋 裁决理由：”或“⚠️ 风险说明”标题，完全信任 AI 的输出已包含这些
        parts = [
            title,
            "",
            execution_block,
            "",
            reasoning_raw,
            "",
            risk_raw
        ]
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
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")
    title = f"策略信号：{symbol}｜⚡ 风控审计 · {now}"
    report = reviewer_report.get("full_report", "无审查报告")
    severity = reviewer_report.get("severity_counts", {})
    summary = f"🔍 审计发现：严重问题{severity.get('高', 0)}个，中等问题{severity.get('中', 0)}个，轻微问题{severity.get('低', 0)}个"
    report = report.replace('【风控审计官 - 审计报告】', '').strip()
    return f"{title}\n\n{summary}\n\n📋 风控审计官 - 审计报告\n> {report}"


def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> str:
    return format_strategy_message(symbol, strategy, data)


# ===== 新增：委员会最终裁决推送模板（按原始块展示，关键字加粗） =====
def format_final_decision(symbol: str, strategy: dict, judge_result: dict = None) -> str:
    """委员会最终裁决推送，直接展示法官C原始区块，并对裁决理由中的关键字加粗"""
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    direction = strategy.get("direction", "neutral")
    size = strategy.get("position_size", "none")
    conf = strategy.get("confidence", "medium")
    verdict = strategy.get("_review_verdict", "")

    size_map = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}
    conf_map = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}
    status_tag_map = {
        "维持原判": "✅审查确认",
        "修正参数": "🔧审查修正",
        "降级执行": "⚠️降级执行",
        "推翻改为观望": "🔄推翻→观望",
        "推翻改为反向操作": "🔄推翻→反向操作"
    }

    if direction == "long":
        dir_icon, dir_text = "🟢", "做多"
    elif direction == "short":
        dir_icon, dir_text = "🔴", "做空"
    else:
        dir_icon, dir_text = "⚪", "观望"

    status_tag = status_tag_map.get(verdict, "")
    title = f"策略信号：{symbol}｜📋 交易委员会：{dir_icon} {dir_text} · {size_map.get(size, '')} · {conf_map.get(conf, '')} · {now} {status_tag}"

    # 获取原始块
    title_line = strategy.get("_title_line", "")
    exec_block = strategy.get("_exec_block_raw", "")
    reasoning_block = strategy.get("_reasoning_block_raw", "")
    risk_block = strategy.get("_risk_block_raw", "")

    # 处理裁决理由：转义特殊字符（除了我们手动加粗的部分），并对关键字加粗
    # 先转义星号和下划线，避免意外解析
    def escape_non_bold(text: str) -> str:
        # 保护已有的加粗标记（**...**）
        # 简单做法：先替换掉所有 * 为 \*，然后再将 \*\* 还原为 **
        text = text.replace('*', r'\*')
        text = text.replace(r'\*\*', '**')  # 恢复手动加粗
        text = text.replace('_', r'\_')
        return text

    # 关键字列表
    keywords = ["指控内容", "裁决结论", "核验依据", "反证风险评估", "核心逻辑"]
    for kw in keywords:
        # 使用正则忽略已经加粗的情况
        reasoning_block = re.sub(
            r'(?<!\*)(' + re.escape(kw) + r')(?!\*)',
            r'**\1**',
            reasoning_block
        )

    # 转义除关键字外的特殊字符
    reasoning_block = escape_non_bold(reasoning_block)

    # 执行指令和风险说明简单转义
    exec_block = exec_block.replace('*', r'\*').replace('_', r'\_')
    risk_block = risk_block.replace('*', r'\*').replace('_', r'\_')
    title_line = title_line.replace('*', r'\*').replace('_', r'\_')

    # 拼装最终消息
    parts = [title]
    if title_line:
        parts.append(title_line)
    if exec_block:
        parts.append(exec_block)
    if reasoning_block:
        parts.append(reasoning_block)
    if risk_block:
        parts.append(risk_block)

    return '\n\n'.join(parts)
