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

# ========== 基础发送（Markdown 单条） ==========
def send_dingtalk_message(content: str, title: str = "策略推送") -> bool:
    """发送一条 Markdown 消息，长度不应超过 4096 字符"""
    webhook = os.getenv("DINGTALK_WEBHOOK_URL", "")
    secret = os.getenv("DINGTALK_SECRET", "")
    if not webhook:
        logger.error("未配置钉钉 Webhook")
        return False

    ts = str(round(time.time() * 1000))
    if secret and secret.lower() != "none":
        try:
            sign_str = f"{ts}\n{secret}"
            signature = base64.b64encode(
                hmac.new(
                    secret.encode("utf-8"),
                    sign_str.encode("utf-8"),
                    hashlib.sha256,
                ).digest()
            ).decode()
            sign = urllib.parse.quote(signature, safe="")  # 优化点1
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
    """从完整推演中提取第七步：制定交易计划（优化点2）"""
    if not text:
        return ""

    # 非贪婪匹配，到下一个“第X步”或文本结束
    pattern = r'第七步[：:]\s*[^\n]*\n(.*?)(?=\n第[一二三四五六七八九十]+\s*步|\Z)'
    m = re.search(pattern, text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        if content:
            return content

    # 退化为取第七步标题行之后全部
    m = re.search(r'(第七步[：:].*)', text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        if content:
            return content

    # 兜底：最后 2000 字符
    return text[-2000:].lstrip('\n')


def _safe_code_block(text: str) -> str:
    """替换掉文本中的三个反引号，防止破坏 Markdown 代码块（优化点3）"""
    if not text:
        return ""
    return text.replace("```", "'''")


# ========== 超长内容自动拆分推送（用于带代码块的消息） ==========
DINGTALK_MAX_CHARS = 4000  # 留出缓冲

def _send_long_with_code_block(body: str, title: str) -> bool:
    """
    用于处理“摘要 + 代码块”结构的超长 Markdown：
    - 将摘要（代码块之前的部分）和代码块拆开
    - 摘要单独发送
    - 代码块内容完整展示，若内容过长则自动分段并用 ``` 包裹
    """
    if len(body) <= DINGTALK_MAX_CHARS:
        return send_dingtalk_message(body, title)

    # 查找第一个代码块起始位置
    code_start = body.find("```")
    if code_start == -1:
        # 没有代码块，按纯文本分段
        return _send_long_fallback(body, title)

    # 分离摘要和代码块部分
    before = body[:code_start].rstrip()
    after_start = body[code_start + 3:]  # 跳过开头的 ```

    # 查找代码块结束标记（下一个 ```）
    code_end = after_start.find("```")
    if code_end == -1:
        # 未闭合，也按普通文本处理
        return _send_long_fallback(body, title)

    code_content = after_start[:code_end].strip()
    after_code = after_start[code_end + 3:].strip()

    # 判断是否因为代码块太长导致整体超限（通常如此）
    # 先发摘要部分
    if before:
        # 在摘要末尾提示“详细推演见下一条”
        hint = "\n\n*（详细推演见下一条消息）*"
        ok = send_dingtalk_message(before + hint, title)
        if not ok:
            return False
        time.sleep(0.6)  # 钉钉频率限制

    # 再发代码块部分
    return _send_codeblock_split(code_content, after_code, title)


def _send_codeblock_split(code_content: str, after_text: str, title: str) -> bool:
    """将代码块内容分段发送，每段都用 ``` 包裹，并添加页码"""
    # 代码块分片，留出页脚空间
    page_footer = "\n\n*（代码内容 {}/{}）*"
    # 估算可用长度（去掉包裹的 ``` 和页脚）
    wrap_len = 6  # 开头 ```\n 和结尾 \n```
    max_chunk = DINGTALK_MAX_CHARS - len(page_footer.format(99, 99)) - wrap_len - 10

    # 如果代码内容本身很短（但前后还可能有 after_text）
    full_block = f"```\n{code_content}\n```"
    if after_text:
        full_block += f"\n{after_text}"

    total_len = len(full_block)
    if total_len <= DINGTALK_MAX_CHARS:
        return send_dingtalk_message(full_block, title)

    # 需要分段：只对 code_content 进行切割，after_text 最后附加
    chunks = []
    remaining = code_content
    while remaining:
        if len(remaining) <= max_chunk:
            chunks.append(remaining)
            break
        # 找一个合适的分割点（优先在换行处）
        cut = max_chunk
        # 往回找最近的 \n
        pos = remaining.rfind('\n', 0, max_chunk)
        if pos > max_chunk // 2:
            cut = pos
        else:
            # 次优：按空格或强制切割
            pos2 = remaining.rfind(' ', 0, max_chunk)
            if pos2 > max_chunk // 2:
                cut = pos2
        chunk = remaining[:cut].rstrip()
        chunks.append(chunk)
        remaining = remaining[cut:].lstrip('\n')

    total_parts = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        footer = page_footer.format(i, total_parts)
        msg = f"```\n{chunk}\n```{footer}"
        if i == total_parts and after_text:
            msg += f"\n{after_text}"
        ok = send_dingtalk_message(msg, f"{title} (推演代码 {i}/{total_parts})")
        if not ok:
            return False
        if i < total_parts:
            time.sleep(0.6)
    return True


def _send_long_fallback(body: str, title: str) -> bool:
    """普通超长文本的简单切片（几乎不会触发）"""
    page_footer = "\n\n*（续 {}/{}）*"
    max_chunk = DINGTALK_MAX_CHARS - len(page_footer.format(99, 99)) - 10

    chunks = []
    remaining = body
    while remaining:
        if len(remaining) <= max_chunk:
            chunks.append(remaining)
            break
        cut = remaining.rfind('\n', 0, max_chunk)
        if cut == -1:
            cut = max_chunk
        chunk = remaining[:cut].rstrip()
        chunks.append(chunk)
        remaining = remaining[cut:].lstrip('\n')

    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        msg = chunk + page_footer.format(i, total)
        ok = send_dingtalk_message(msg, f"{title} ({i}/{total})")
        if not ok:
            return False
        if i < total:
            time.sleep(0.6)
    return True


# ========== 1. 首席交易员信号 ==========
def format_strategy_message(symbol: str, strategy: dict, data: dict) -> bool:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%m-%d %H:%M")

    direction = strategy.get("direction", "neutral")
    pos_size = strategy.get("position_size", "none")
    conf = strategy.get("confidence", "medium")
    entry_low = strategy.get("entry_price_low", 0) or 0
    entry_high = strategy.get("entry_price_high", 0) or 0
    stop_loss = strategy.get("stop_loss", 0) or 0
    take_profit = strategy.get("take_profit", 0) or 0
    current = (data.get("mark_price", 0) or 0) if data else 0

    dir_icon = {"long": "🟢", "short": "🔴", "neutral": "⚪"}.get(direction, "⚪")
    dir_text = {"long": "做多", "short": "做空", "neutral": "观望"}.get(direction, "观望")
    size_text = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}.get(pos_size, "?")
    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "?")

    header = f"### 🧠 首席交易员 · 提交审计 | {symbol} | {dir_icon} {dir_text} | {size_text} | 置信度{conf_icon} | {now}"
    price_block = (
        f"现价 {current:.0f}  |  "
        f"入场 {entry_low:.0f}-{entry_high:.0f}  |  "
        f"止损 {stop_loss:.0f}  |  "
        f"止盈 {take_profit:.0f}"
    )

    full_reasoning = strategy.get("reasoning", "")
    final_step = _extract_final_step(full_reasoning)
    if not final_step:
        final_step = "未解析到第七步"

    # 使用代码块包裹完整第七步内容
    body = (
        f"{header}\n"
        f"{price_block}\n\n"
        f"**第七步 · 交易计划**\n"
        f"```\n{_safe_code_block(final_step)}\n```"
    )

    return _send_long_with_code_block(body, f"首席交易员·{symbol}")


# ========== 2. 风控审计报告（暂时不变，也用分段保护） ==========
def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> bool:
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
    return _send_long_with_code_block(body, f"风控审计·{symbol}")


# ========== 3. 交易委员会最终裁决（暂时不变） ==========
def format_final_decision(symbol: str, strategy: dict, judge_result: dict = None, data: dict = None) -> bool:
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
    return _send_long_with_code_block(body, f"最终裁决·{symbol}")


# ========== 兼容旧版 ==========
def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> bool:
    return format_final_decision(symbol, strategy, judge_result, data)
