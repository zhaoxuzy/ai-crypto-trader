# notifier/dingtalk.py
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
        try:
            sign_str = f"{ts}\n{secret}"
            # 优化点 1：使用 quote 替代 quote_plus，避免 + 号引发验签失败
            signature = base64.b64encode(
                hmac.new(
                    secret.encode("utf-8"),
                    sign_str.encode("utf-8"),
                    hashlib.sha256,
                ).digest()
            ).decode()
            sign = urllib.parse.quote(signature, safe="")
            url_parts = list(urllib.parse.urlparse(webhook))
            query = dict(urllib.parse.parse_qsl(url_parts[4]))
            query.update({"timestamp": ts, "sign": sign})
            url_parts[4] = urllib.parse.urlencode(query)
            webhook = urllib.parse.urlunparse(url_parts)
        except Exception as e:
            logger.error(f"钉钉签名生成失败: {e}")
            return False

    try:
        payload = {"msgtype": "markdown", "markdown": {"title": title, "text": content}}
        resp = requests.post(webhook, json=payload, timeout=10)
        resp_data = resp.json()
        if resp_data.get("errcode") == 0:
            logger.info("钉钉推送成功")
            return True
        logger.error(f"钉钉失败: {resp_data}")
        return False
    except Exception as e:
        logger.error(f"钉钉异常: {e}")
        return False


# ========== 工具函数 ==========
def _extract_final_step(text: str) -> str:
    """从完整推演中提取第七步：制定交易计划（优化点 2）"""
    if not text:
        return ""

    # 非贪婪匹配“第七步：/：”后的内容，直到下一个“第X步”或文本结束
    pattern = r'第七步[：:]\s*[^\n]*\n(.*?)(?=\n第[一二三四五六七八九十]+\s*步|\Z)'
    m = re.search(pattern, text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        if content:
            return content

    # 如果没找到下一个步骤，退化为取第七步标题行后所有内容
    m = re.search(r'(第七步[：:].*)', text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        if content:
            return content

    # 最后兜底：取最后 2000 字符
    return text[-2000:].lstrip('\n')


def _safe_code_block(text: str) -> str:
    """将文本中的三个反引号暂时替换，避免破坏 Markdown 代码块（优化点 3）"""
    if not text:
        return ""
    return text.replace("```", "'''")


# ========== 1. 首席交易员初步信 ==========
def format_strategy_message(symbol: str, strategy: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    direction = strategy.get("direction", "neutral")
    pos_size = strategy.get("position_size", "none")
    conf = strategy.get("confidence", "medium")
    entry_low = strategy.get("entry_price_low", 0) or 0
    entry_high = strategy.get("entry_price_high", 0) or 0
    stop_loss = strategy.get("stop_loss", 0) or 0
    take_profit = strategy.get("take_profit", 0) or 0
    current = data.get("mark_price", 0) or 0

    dir_icon = {"long": "🟢", "short": "🔴", "neutral": "⚪"}.get(direction, "⚪")
    dir_text = {"long": "做多", "short": "做空", "neutral": "观望"}.get(direction, "观望")
    size_text = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}.get(pos_size, "?")
    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "?")

    header = f"### 🧠 首席交易员 · 提交审计 | {symbol} | {dir_icon} {dir_text} | {size_text} | 置信度{conf_icon} | {now}"

    if direction == "neutral":
        price_block = "**方向：观望，暂不设置入场/止损/止盈**"
    else:
        price_block = (
            f"现价 {current:.0f}  |  "
            f"入场 {entry_low:.0f}-{entry_high:.0f}  |  "
            f"止损 {stop_loss:.0f}  |  "
            f"止盈 {take_profit:.0f}"
        )

    # 改回显示完整的推理过程，不做提取第七步
    reasoning = strategy.get("reasoning", "无推演过程")

    # 可选：去除乱码字符，防止出现这类怪符号
    reasoning = reasoning.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

    body = (
        f"{header}\n"
        f"{price_block}\n\n"
        f"**推演过程**\n"                    # 恢复为「推演过程」
        f"````\n{reasoning}\n````"
    )
    return body


# ========== 2. 风控审计报告 ==========
def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    severity = reviewer_report.get("severity_counts", {"高": 0, "中": 0, "低": 0})
    high = severity.get("高", 0)
    medium = severity.get("中", 0)
    low = severity.get("低", 0)

    if high > 0:
        conclusion = "⛔ 驳回"
    elif medium > 0 or low > 0:
        conclusion = "⚠️ 存疑"
    else:
        conclusion = "✅ 通过"

    header = f"### ⚡ 风控审计官 · 审计完成 | {symbol} | {conclusion} | {now}"
    sev_line = f"严重 {high}  中等 {medium}  轻微 {low}"

    report = reviewer_report.get("full_report", "无审查报告")

    body = (
        f"{header}\n"
        f"{sev_line}\n\n"
        f"**审计报告**\n"
        f"```\n{_safe_code_block(report)}\n```"
    )
    return body


# ========== 3. 交易委员会最终裁决 ==========
def format_final_decision(symbol: str, strategy: dict, judge_result: dict = None, data: dict = None) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    direction = strategy.get("direction", "neutral")
    pos_size = strategy.get("position_size", "none")
    conf = strategy.get("confidence", "medium")
    verdict = strategy.get("_review_verdict", "")
    entry_low = strategy.get("entry_price_low", 0) or 0
    entry_high = strategy.get("entry_price_high", 0) or 0
    stop_loss = strategy.get("stop_loss", 0) or 0
    take_profit = strategy.get("take_profit", 0) or 0
    current = (data.get("mark_price", 0) or 0) if data else 0

    dir_icon = {"long": "🟢", "short": "🔴", "neutral": "⚪"}.get(direction, "⚪")
    dir_text = {"long": "做多", "short": "做空", "neutral": "观望"}.get(direction, "观望")
    size_text = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}.get(pos_size, "?")
    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "?")

    verdict_map = {
        "维持原判": "✅维持原判",
        "推翻改为观望": "🔄推翻→观望",
        "推翻改为反向操作": "🔄推翻",
    }
    verdict_text = verdict_map.get(verdict, verdict) if verdict else "裁决"
    verdict_icon = "✅" if verdict == "维持原判" else "🔄"

    header = f"### 📋 交易委员会 · 最终裁决 | {symbol} | {verdict_icon} {verdict_text} | {now}"
    decision_line = f"{dir_icon} {dir_text} | {size_text} | 置信度{conf_icon}"
    price_block = (
        f"现价 {current:.1f}  |  "
        f"入场 {entry_low:.0f}-{entry_high:.0f}  |  "
        f"止损 {stop_loss:.0f}  |  "
        f"止盈 {take_profit:.0f}"
    )

    judge_content = strategy.get("_judge_reasoning", "")
    if not judge_content:
        judge_content = "无裁决内容"

    body = (
        f"{header}\n"
        f"{decision_line}\n"
        f"{price_block}\n\n"
        f"**裁决内容**\n"
        f"```\n{_safe_code_block(judge_content)}\n```"
    )
    return body


# ========== 兼容旧版 ==========
def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> str:
    return format_final_decision(symbol, strategy, judge_result, data)
