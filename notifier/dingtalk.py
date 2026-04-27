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
        # 构建标题与状态标签
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

        # 价格参数
        entry_low = strategy.get("entry_price_low", 0)
        entry_high = strategy.get("entry_price_high", 0)
        stop = strategy.get("stop_loss", 0)
        tp = strategy.get("take_profit", 0)
        current = data.get("mark_price", 0)

        # 方向图标
        dir_icon = "⚪" if direction == "neutral" else ("🟢" if direction == "long" else "🔴")
        dir_text = "观望" if direction == "neutral" else ("做多" if direction == "long" else "做空")

        # 入场、止损、止盈显示
        if direction == "neutral":
            entry_str = "无"
            stop_str = "无"
            tp_str = "无"
        else:
            entry_str = f"{entry_low:.0f}-{entry_high:.0f}"
            stop_str = f"{stop:.0f}"
            tp_str = f"{tp:.0f}"

        # 说明（执行指令）
        explanation = strategy.get("execution_plan", "")

        # 最终策略块
        final_strategy_block = (
            f"📋 最终策略\n"
            f"•   方向：{dir_icon} {dir_text}\n"
            f"•   现价：{current:.0f}\n"
            f"•   入场区间：{entry_str}\n"
            f"•   止损：{stop_str}\n"
            f"•   止盈：{tp_str}\n"
            f"•   说明：{explanation}"
        )

        # 裁决理由：取 _judge_reasoning 并清理
        reasoning_text = strategy.get("_judge_reasoning", "")
        # 清理可能残留的标签
        reasoning_text = re.sub(r'📋\s*裁决理由[：:]?', '', reasoning_text).strip()
        reasoning_block = f"📋 裁决理由：\n{reasoning_text}" if reasoning_text else ""

        # 风险说明
        risk_note = strategy.get("risk_note", "请严格设置止损")
        risk_block = f"⚠️ 风险说明\n{risk_note}"
        # 如果是推翻或驳回，追加决议说明
        if verdict in ["推翻改为观望", "推翻改为反向操作"]:
            risk_block += f"\n\n交易委员会决议: {verdict}"

        # 拼接最终消息
        parts = [title, "", final_strategy_block]
        if reasoning_block:
            parts.append(reasoning_block)
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
