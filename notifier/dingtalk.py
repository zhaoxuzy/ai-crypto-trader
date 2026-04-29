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
from typing import Dict, Optional
from utils.logger import logger

# ===================== 基础推送 =====================
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
            signature = base64.b64encode(
                hmac.new(secret.encode(), sign_str.encode(), hashlib.sha256).digest()
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

# ===================== 文本提取 =====================
def _extract_final_step(text: str) -> str:
    """提取首席交易员第七步"""
    if not text:
        return ""
    # 非贪婪到下一个“第X步”
    pattern = r'第七步[：:]\s*[^\n]*\n(.*?)(?=\n第[一二三四五六七八九十]+\s*步|\Z)'
    m = re.search(pattern, text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # 回退
    m = re.search(r'(第七步[：:].*)', text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return text[-2000:].lstrip('\n')

def _safe_code_block(text: str) -> str:
    """避免代码块内部的三反引号破坏结构"""
    if not text:
        return ""
    return text.replace("```", "'''")

# ===================== 裁决文本解析 =====================
def _parse_judge_execution(text: str) -> Dict[str, Optional[str]]:
    """
    从裁决全文中提取最终判决和执行指令字段
    返回包含:
      verdict: "维持原判" or "推翻"
      direction, position_size, entry_low, entry_high, stop_loss, take_profit
    若提取失败，返回空 dict
    """
    if not text:
        return {}

    result = {}

    # 1. 提取最终判决（只保留两种）
    m = re.search(r'📌\s*最终判决[：:]\s*(.*?)(?:\n|$)', text)
    if m:
        raw_verdict = m.group(1).strip()
        if "推翻" in raw_verdict:
            result["verdict"] = "推翻"
        else:
            result["verdict"] = "维持原判"
    else:
        # 尝试兼容无 emoji 的情况
        m2 = re.search(r'最终判决[：:]\s*(.*?)(?:\n|$)', text)
        if m2:
            raw = m2.group(1).strip()
            result["verdict"] = "推翻" if "推翻" in raw else "维持原判"

    # 2. 定位 🎯执行指令 段落
    exec_start = re.search(r'🎯\s*执行指令', text)
    if not exec_start:
        return result  # 没找到，只返回判决结果
    exec_text = text[exec_start.start():]

    # 3. 提取各字段（兼容中文全角/半角冒号，前后空格）
    def _extract(pattern, target, default=None):
        m = re.search(pattern, target)
        if m:
            return m.group(1).strip()
        return default

    result["direction"] = _extract(r'方向[：:]\s*([做多空观望]+)', exec_text, "")
    pos = _extract(r'仓位[：:]\s*([^\n]+)', exec_text, "")
    # 仓位可能包含括号备注，只保留主体
    if pos:
        pos = pos.split("（")[0].split("(")[0].strip()
    result["position_size"] = pos

    # 入场区间：匹配 数字-数字 或 数字~数字
    entry_match = re.search(r'入场区间[：:]\s*([\d.,]+)\s*[-~至]+\s*([\d.,]+)', exec_text)
    if entry_match:
        try:
            result["entry_low"] = float(entry_match.group(1).replace(",", ""))
            result["entry_high"] = float(entry_match.group(2).replace(",", ""))
        except ValueError:
            pass

    # 止损
    sl_match = re.search(r'止损[：:]\s*([\d.,]+)', exec_text)
    if sl_match:
        try:
            result["stop_loss"] = float(sl_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # 止盈
    tp_match = re.search(r'止盈[：:]\s*([\d.,]+)', exec_text)
    if tp_match:
        try:
            result["take_profit"] = float(tp_match.group(1).replace(",", ""))
        except ValueError:
            pass

    return result

# ===================== 超长拆分发送 =====================
DINGTALK_MAX_CHARS = 4000

def _send_long_with_code_block(body: str, title: str) -> bool:
    """处理带代码块的消息自动拆分"""
    if len(body) <= DINGTALK_MAX_CHARS:
        return send_dingtalk_message(body, title)

    # 查找第一个 ```
    code_start = body.find("```")
    if code_start == -1:
        return _send_long_fallback(body, title)

    before = body[:code_start].rstrip()
    after_start = body[code_start+3:]
    code_end = after_start.find("```")
    if code_end == -1:
        return _send_long_fallback(body, title)

    code_content = after_start[:code_end].strip()
    after_code = after_start[code_end+3:].strip()

    # 发送摘要
    if before:
        if not send_dingtalk_message(before + "\n\n*（详细内容见下一条）*", title):
            return False
        time.sleep(0.6)

    return _send_codeblock_split(code_content, after_code, title)

def _send_codeblock_split(code_content: str, after_text: str, title: str) -> bool:
    page_footer = "\n\n*（推演 {}/{}）*"
    max_chunk = DINGTALK_MAX_CHARS - len(page_footer.format(99,99)) - 10

    full_block = f"```\n{code_content}\n```"
    if after_text:
        full_block += f"\n{after_text}"
    if len(full_block) <= DINGTALK_MAX_CHARS:
        return send_dingtalk_message(full_block, title)

    # 拆分代码内容
    chunks = []
    remaining = code_content
    while remaining:
        if len(remaining) <= max_chunk:
            chunks.append(remaining)
            break
        cut_pos = remaining.rfind('\n', 0, max_chunk)
        if cut_pos == -1 or cut_pos < max_chunk//2:
            cut_pos = remaining.rfind(' ', 0, max_chunk)
        if cut_pos == -1 or cut_pos < max_chunk//2:
            cut_pos = max_chunk
        chunks.append(remaining[:cut_pos].rstrip())
        remaining = remaining[cut_pos:].lstrip('\n')

    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        msg = f"```\n{chunk}\n```" + page_footer.format(i, total)
        if i == total and after_text:
            msg += f"\n{after_text}"
        if not send_dingtalk_message(msg, f"{title}({i}/{total})"):
            return False
        if i < total:
            time.sleep(0.6)
    return True

def _send_long_fallback(body: str, title: str) -> bool:
    page_footer = "\n
