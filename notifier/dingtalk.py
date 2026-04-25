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

# 钉钉 Markdown 消息最大长度（官方限制约20000，我们留出安全边界）
DINGTALK_MAX_CONTENT_LENGTH = 18000


def send_dingtalk_message(content: str, title: str = "策略推送") -> bool:
    """
    发送钉钉消息，如果内容超过长度限制，自动拆分成多条发送。
    """
    webhook = get_webhook()
    if not webhook:
        return False

    # 如果内容不超长，直接发送
    if len(content) <= DINGTALK_MAX_CONTENT_LENGTH:
        return _post_dingtalk(webhook, content, title)

    # 需要拆分发送
    logger.info(f"消息长度 {len(content)} 超出限制，将拆分为多条发送")
    parts = split_long_message(content)
    success = True
    for i, part in enumerate(parts):
        part_title = f"{title} ({i+1}/{len(parts)})" if len(parts) > 1 else title
        if not _post_dingtalk(webhook, part, part_title):
            success = False
            logger.error(f"拆分消息第 {i+1} 条发送失败")
    return success


def get_webhook() -> str:
    """获取带签名的完整 Webhook URL"""
    webhook = os.getenv("DINGTALK_WEBHOOK_URL", "")
    secret = os.getenv("DINGTALK_SECRET", "")
    if not webhook:
        logger.error("未配置钉钉 Webhook")
        return ""
    if secret and secret.lower() != "none":
        ts = str(round(time.time() * 1000))
        sign_str = f"{ts}\n{secret}"
        sign = urllib.parse.quote_plus(
            base64.b64encode(
                hmac.new(secret.encode(), sign_str.encode(), hashlib.sha256).digest()
            )
        )
        webhook = f"{webhook}&timestamp={ts}&sign={sign}"
    return webhook


def _post_dingtalk(webhook: str, content: str, title: str) -> bool:
    """发送单条消息"""
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


def split_long_message(content: str) -> list:
    """
    将长消息按段落拆分，保证每条不超过限制。
    优先在段落边界（\n\n）处切割。
    """
    parts = []
    current_part = ""
    paragraphs = content.split("\n\n")

    for para in paragraphs:
        if len(current_part) + len(para) + 2 <= DINGTALK_MAX_CONTENT_LENGTH:
            if current_part:
                current_part += "\n\n" + para
            else:
                current_part = para
        else:
            if current_part:
                parts.append(current_part)
            if len(para) > DINGTALK_MAX_CONTENT_LENGTH:
                logger.warning("单个段落超长，将强制切割")
                for i in range(0, len(para), DINGTALK_MAX_CONTENT_LENGTH):
                    parts.append(para[i:i + DINGTALK_MAX_CONTENT_LENGTH])
                current_part = ""
            else:
                current_part = para

    if current_part:
        parts.append(current_part)

    return parts


def format_reasoning(text: str) -> str:
    """
    格式化推理文本：
    - 步骤标题加粗
    - 所有次级标签独立成行，前缀为“> ”加两个空格，确保不与上文粘连
    - 强制在次级标签前插入换行符
    """
    if not text:
        return "> 无推理过程"

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 强制在次级标签前插入换行符（避免与上文粘连）
    secondary_labels = [
        "分析数据[：:]", "第一反应[：:]", "自我质疑[：:]", "最终结论[：:]",
        "信号传唤[：:]", "权重审判[：:]", "心证交锋[：:]", "核心假设[：:]", "证伪条件[：:]",
        "价格路径推演[：:]", "合约策略[：:]", "主动证伪信号[：:]", "微观盘口确认[：:]"
    ]
    for label in secondary_labels:
        text = re.sub(rf'(?<!\n)({label})', r'\n\1', text)

    lines = text.split('\n')
    quoted = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            quoted.append('> ')
            continue

        # 步骤标题加粗
        if re.match(r'^第[一二三四五六七八九]步[：:]', stripped_line):
            stripped_line = re.sub(r'^(第[一二三四五六七八九]步)', r'**\1**', stripped_line)
            quoted.append(f'> {stripped_line}')
        
        # 次级标题不加粗，但独立成行，前缀为“> ”加两个空格
        elif re.match(r'^(分析数据|第一反应|自我质疑|最终结论|信号传唤|权重审判|心证交锋|核心假设|证伪条件|价格路径推演|合约策略|主动证伪信号|微观盘口确认)[：:]', stripped_line):
            quoted.append(f'>   {stripped_line}')
        
        # 普通行
        else:
            quoted.append(f'> {stripped_line}')

    # 压缩连续空行
    cleaned = []
    prev_empty = False
    for q in quoted:
        is_empty = (q.strip() == '>' or q.strip() == '')
        if is_empty and prev_empty:
            continue
        cleaned.append(q)
        prev_empty = is_empty
    return '\n'.join(cleaned)


def format_strategy_message(symbol: str, strategy: dict, data: dict) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")
    direction = strategy.get("direction", "neutral")

    # ----- 标题与参数卡片 -----
    if direction == "neutral":
        title_line = f"## ⚪ 观望 {symbol} · 🔴低 · {now}"
        param_card = f"> 现价{data.get('mark_price',0):.0f} · 入场0-0 · 止损0 · 止盈0 · 盈亏比N/A"
    else:
        emoji = "🟢" if direction == "long" else "🔴"
        dir_text = "做多" if direction == "long" else "做空"
        size = strategy.get("position_size", "none")
        size_cn = {"light": "轻仓", "medium": "中仓", "heavy": "重仓"}.get(size, "")
        conf = strategy.get("confidence", "medium")
        conf_cn = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}.get(conf, "🟡中")

        title_parts = [f"{emoji} {dir_text} {symbol}"]
        if size_cn:
            title_parts.append(size_cn)
        title_parts.append(conf_cn)
        title_parts.append(now)
        title_line = "## " + " · ".join(title_parts)

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
        param_card = f"> 现价{current:.0f} · 入场{entry_low:.0f}-{entry_high:.0f} · 止损{stop:.0f} · 止盈{tp:.0f} · 盈亏比{rr_str}"

    # ----- 推理内容 -----
    reasoning_raw = strategy.get("reasoning", "无推理过程")
    reasoning_block = format_reasoning(reasoning_raw)

    # ----- 风险说明 -----
    risk_raw = strategy.get("risk_note", "请严格设置止损")
    risk_lines = [f"> {line.strip()}" for line in risk_raw.split('\n') if line.strip()]
    if not risk_lines:
        risk_lines = ["> 请严格设置止损"]
    risk_block = "> ### ⚠️ 风险说明\n" + "\n".join(risk_lines)

    # ----- 脚注 -----
    atr = data.get("atr_15m", 0)
    funding = data.get("funding_rate", 0)
    oi_chg = data.get("oi_change_24h", 0)
    cvd = data.get("cvd_slope", 0)
    cvd_dir = "↗" if cvd > 0 else ("↘" if cvd < 0 else "→")
    fg = data.get("fear_greed", 50)
    footnote = f"📎 ATR{atr:.0f} · 费率{funding:.4f}% · OI{oi_chg:+.1f}% · CVD{cvd_dir} · 贪婪{fg}"

    # ----- 拼接最终消息 -----
    return f"{title_line}\n\n{param_card}\n\n### 🧠 交易员推理\n{reasoning_block}\n\n{risk_block}\n\n{footnote}"