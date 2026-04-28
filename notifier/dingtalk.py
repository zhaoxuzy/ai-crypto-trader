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


# ========== 基础推送 ==========
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


# ========== 格式化工具 ==========
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
        elif re.match(r'^(分析数据|第一反应|自我质疑|最终结论|交叉验证与裁决|价格路径推演|最终合约策略|推理自检|入场区间|止损位|止盈位|主动证伪信号|微观盘口确认)[：:]', line):
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


# ========== 首席交易员初步信号 ==========
def format_strategy_message(symbol: str, strategy: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")
    direction = strategy.get("direction", "neutral")

    reviewed = strategy.get("_reviewed", False)
    verdict = strategy.get("_review_verdict", "")
    preliminary = strategy.get("_preliminary", False)

    # --- 审查后最终信号 ---
    if reviewed and not preliminary:
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

        exec_plan = strategy.get("execution_plan", "无具体指令")
        execution_block = f"🎯 执行指令\n{exec_plan}"
        reasoning_raw = strategy.get("_judge_reasoning", "")
        if not reasoning_raw.strip():
            reasoning_raw = "无裁决理由"
        risk_raw = strategy.get("risk_note", "请严格设置止损")

        return '\n\n'.join([title, "", execution_block, "", reasoning_raw, "", risk_raw])

    # --- 初步信号（审查中） ---
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


# ========== 风控审计报告 ==========
def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")
    title = f"策略信号：{symbol}｜⚡ 风控审计 · {now}"
    report = reviewer_report.get("full_report", "无审查报告")
    severity = reviewer_report.get("severity_counts", {})
    summary = f"🔍 审计发现：严重问题{severity.get('高', 0)}个，中等问题{severity.get('中', 0)}个，轻微问题{severity.get('低', 0)}个"

    report = report.replace('【风控审计官 - 审计报告】', '').strip()
    report = re.sub(r'^---\s*', '', report, flags=re.MULTILINE)
    report = re.sub(r'(?<!\n)(?=[一二三四五六七八九十]、)', '\n\n', report)
    report = re.sub(r'(?<!\n)(?=[-•])', '\n', report)
    report = report.strip()

    # 标题后增加空行，确保“一、”独立成行
    return f"{title}\n\n{summary}\n\n📋 风控审计官 - 审计报告\n\n{report}"


def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> str:
    return format_strategy_message(symbol, strategy, data)


# ========== 交易委员会最终裁决（更新版） ==========
def format_final_decision(symbol: str, strategy: dict, judge_result: dict = None, data: dict = None) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    direction = strategy.get("direction", "neutral")
    conf = strategy.get("confidence", "medium")
    verdict = strategy.get("_review_verdict", "")

    # 中英文仓位映射
    size_map = {
        "light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无",
        "轻仓": "轻仓", "中仓": "中仓", "重仓": "重仓", "无": "无"
    }
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

    # 获取原始块（优先从 strategy，否则从 judge_result）
    exec_block = strategy.get("_exec_block_raw", "")
    if not exec_block and judge_result:
        exec_block = judge_result.get("judge_C", {}).get("exec_block", "")
    title_line = strategy.get("_title_line", "")
    reasoning_block = strategy.get("_reasoning_block_raw", "")
    risk_block = strategy.get("_risk_block_raw", "")

    # ----- 提取仓位，支持中英文，且优先从 judge_result 中取正确的英文值 -----
    size = "无仓位"
    if exec_block:
        m = re.search(r'仓位[：:]\s*(\S+)', exec_block)
        if m:
            raw = m.group(1).strip()
            size = size_map.get(raw, raw)
    if size == "无仓位" and judge_result:
        # 如果 block 中没找到，尝试用 judge_C 里的 final_position_size
        pos = judge_result.get("judge_C", {}).get("final_position_size", "")
        if pos:
            size = size_map.get(pos, pos)

    title = f"策略信号：{symbol}｜📋 交易委员会：{dir_icon} {dir_text} · {size} · {conf_map.get(conf, '')} · {now} {status_tag}"

    # 清理代码块
    def _remove_code_blocks(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'```', '', text)
        return text

    title_line = _remove_code_blocks(title_line)
    exec_block = _remove_code_blocks(exec_block)
    reasoning_block = _remove_code_blocks(reasoning_block)
    risk_block = _remove_code_blocks(risk_block)

    # 转义（保留手工加粗）
    def escape_non_bold(text: str) -> str:
        text = text.replace('*', r'\*')
        text = text.replace(r'\*\*', '**')
        text = text.replace('_', r'\_')
        return text

    keywords = ["指控内容", "裁决结论", "核验依据", "反证风险评估", "核心逻辑"]
    for kw in keywords:
        reasoning_block = re.sub(
            r'(?<!\*)(' + re.escape(kw) + r')(?!\*)',
            r'**\1**',
            reasoning_block
        )

    reasoning_block = escape_non_bold(reasoning_block)
    exec_block = exec_block.replace('*', r'\*').replace('_', r'\_')
    risk_block = risk_block.replace('*', r'\*').replace('_', r'\_')
    title_line = title_line.replace('*', r'\*').replace('_', r'\_')

    # ----- 在仓位行之后插入现价，兼容多种缩进和符号（- • · 等） -----
    if data and data.get("mark_price", 0) > 0 and exec_block:
        price = data["mark_price"]
        if '现价' not in exec_block:
            # 匹配行首缩进 + 符号(-或•或·) + “仓位：” 或 “仓位:”
            exec_block = re.sub(
                r'(^\s*[-•·]\s*仓位[：:][^\n]*)',
                r'\1\n   - 现价：{:.1f}'.format(price),
                exec_block,
                count=1,
                flags=re.MULTILINE
            )

    # 拼装消息
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
