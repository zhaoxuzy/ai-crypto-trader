# notifier/dingtalk.py 重构版
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


# ========== 基础发送函数 (不变) ==========
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


# ========== 工具函数 ==========
def _safe(s: str) -> str:
    """转义钉钉 markdown 中的特殊字符，避免乱码"""
    if not s:
        return ""
    s = s.replace('\\', '\\\\')  # 反斜杠必须最先转义
    s = s.replace('_', '\\_')
    s = s.replace('*', '\\*')
    s = s.replace('`', '\\`')
    s = s.replace('[', '\\[')
    s = s.replace(']', '\\]')
    return s


def _bold(text: str) -> str:
    """将文本包在加粗符号中"""
    return f"**{text}**"


def _now():
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%m-%d %H:%M")


# ========== 1. 首席交易员初步信号 (审查中) ==========
def format_strategy_message(symbol: str, strategy: dict, data: dict) -> str:
    """
    初步信号：只发送一次，展示核心参数、价格推演和风险。
    当 strategy['_preliminary'] 为 True 时，使用审查中模板。
    否则（例如超时降级）回退到旧格式，保证兼容性。
    """
    if strategy.get("_preliminary"):
        # ----- 审查中样式 -----
        direction = strategy.get("direction", "neutral")
        pos_size = strategy.get("position_size", "none")
        confidence = strategy.get("confidence", "medium")

        # 方向映射
        if direction == "long":
            dir_icon, dir_text = "🟢", "做多"
        elif direction == "short":
            dir_icon, dir_text = "🔴", "做空"
        else:
            dir_icon, dir_text = "⚪", "观望"

        size_map = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}
        conf_map = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}

        title = _bold(f"策略信号：{symbol} 🔍审查中…")
        meta = (f"{dir_icon} {dir_text} · {size_map.get(pos_size, '未知')} · {conf_map.get(confidence, '中')}\n"
                f"🧠首席交易员 ⏱ {_now()}")

        # 价格参数
        entry_low = strategy.get("entry_price_low", 0)
        entry_high = strategy.get("entry_price_high", 0)
        stop_loss = strategy.get("stop_loss", 0)
        take_profit = strategy.get("take_profit", 0)
        current = data.get("mark_price", strategy.get("mark_price", 0))
        rr = strategy.get("_calculated_rr", 0)
        rr_str = f"{rr:.2f}" if rr else "N/A"

        price_line = (f"📊 现价 {current:.0f} · 入场 {entry_low:.0f}-{entry_high:.0f} · "
                      f"止损 {stop_loss:.0f} · 止盈 {take_profit:.0f} · 盈亏比 {rr_str}")

        # 价格路径推演 (从 reasoning 末尾提取，或使用第七步总结)
        # 如果 model 在 JSON 中有 execution_plan 也可以使用
        exec_plan = strategy.get("execution_plan", "")
        if not exec_plan:
            # 尝试从 reasoning 最后一段提取第七步
            reasoning = strategy.get("reasoning", "")
            # 简单提取最后一句作为推演路径
            lines = reasoning.strip().split('\n')
            # 找包含“第七步”或“价格路径”的行
            path_line = ""
            for i, line in enumerate(lines):
                if "第七步" in line or "价格路径" in line:
                    path_line = " ".join(lines[i:]).strip()
                    break
            if not path_line and lines:
                # 取最后200字符
                path_line = lines[-1] if len(lines[-1]) > 20 else " ".join(lines[-3:])
            exec_plan = path_line if path_line else "等待信号确认"
        path_text = _safe(exec_plan[:300])  # 截断防止过长

        risk = strategy.get("risk_note", "请严格设置止损")
        risk = _safe(risk)

        msg = (f"{title}\n{meta}\n\n"
               f"{price_line}\n\n"
               f"🗺️ 价格路径推演：{path_text}\n\n"
               f"⚠️ 风险说明：{risk}")
        return msg

    else:
        # ----- 旧版备选 (超时降级等情况) -----
        # 保留原先逻辑，避免出错
        from datetime import datetime as dt, timezone as tz, timedelta
        now = datetime.now(timezone(timedelta(hours=8))).strftime("%m-%d %H:%M")
        direction = strategy.get("direction", "neutral")
        if direction == "neutral":
            title = f"策略信号：{symbol}｜📋 交易委员会：⚪ 观望 · {now} · ⚠️风控驳回"
        else:
            emoji = "🟢" if direction == "long" else "🔴"
            text = "做多" if direction == "long" else "做空"
            size = strategy.get("position_size", "none")
            size_cn = {"light": "轻仓", "medium": "中仓", "heavy": "重仓"}.get(size, "")
            conf = strategy.get("confidence", "medium")
            conf_cn = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}.get(conf, "🟡中")
            verdict = strategy.get("_review_verdict", "")
            status_tag = ""
            if verdict == "推翻改为反向操作":
                status_tag = " · 🔄推翻"
            title = f"策略信号：{symbol}｜📋 交易委员会：{emoji} {text} · {size_cn} · {conf_cn} · {now}{status_tag}"

        exec_plan = strategy.get("execution_plan", "无具体指令")
        reasoning_raw = strategy.get("_judge_reasoning", strategy.get("reasoning", ""))
        risk_raw = strategy.get("risk_note", "请严格设置止损")

        parts = [title, "", f"🎯 执行指令\n{exec_plan}", "", reasoning_raw, "", risk_raw]
        return '\n\n'.join(parts)


# ========== 2. 风控审计报告 (简要版) ==========
def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict) -> str:
    counts = reviewer_report.get("severity_counts", {"高": 0, "中": 0, "低": 0})
    high = counts.get("高", 0)
    medium = counts.get("中", 0)
    low = counts.get("低", 0)

    # 审计结论
    if high > 0:
        conclusion = "❌ 驳回"
    elif medium > 0 or low > 0:
        conclusion = "⚠️ 存疑"
    else:
        conclusion = "✅ 通过"

    # 提取核心错误描述（取 full_report 中前两个严重项）
    report_text = reviewer_report.get("full_report", "")
    core_errors = []
    for line in report_text.split('\n'):
        if line.strip().startswith('-') and '严重性：高' in line:
            # 简化显示
            core_errors.append(line.strip('- ').strip())
        if line.strip().startswith('-') and '严重性：中' in line:
            core_errors.append(line.strip('- ').strip())
        if len(core_errors) >= 2:
            break

    error_brief = "；".join(core_errors) if core_errors else "无关键错误"

    # 反方提示 (从第四部分提取)
    anti_text = ""
    for line in report_text.split('\n'):
        if "做市商" in line or "反向" in line or "猎杀" in line:
            anti_text = line.strip()
            break
    if not anti_text:
        anti_text = "请关注清算池博弈"

    title = _bold(f"策略信号：{symbol} ⚡审计完成")
    meta = (f"🔴严重{high} · 🟡中等{medium} · ⚪轻微{low}\n"
            f"⚡风控审计官 ⏱ {_now()}")

    msg = (f"{title}\n{meta}\n\n"
           f"📋 审计结论：{conclusion} ({_safe(error_brief)})\n"
           f"⚖️ 反方提示：{_safe(anti_text)}")
    return msg


# ========== 3. 交易委员会最终裁决 (简要版) ==========
def format_final_decision(symbol: str, strategy: dict, judge_result: dict = None) -> str:
    """
    最终裁决，展示执行指令、核心逻辑和风险。
    """
    direction = strategy.get("direction", "neutral")
    pos_size = strategy.get("position_size", "none")
    confidence = strategy.get("confidence", "medium")
    verdict = strategy.get("_review_verdict", "")

    # 方向图标
    if direction == "long":
        dir_icon, dir_text = "🟢", "做多"
    elif direction == "short":
        dir_icon, dir_text = "🔴", "做空"
    else:
        dir_icon, dir_text = "⚪", "观望"

    size_map = {"light": "轻仓", "medium": "中仓", "heavy": "重仓", "none": "无仓位"}
    conf_map = {"high": "🟢高", "medium": "🟡中", "low": "🔴低"}

    # 标题
    title = _bold(f"策略信号：{symbol} 📋最终裁决")

    # 第二行：判决动作 + 方向 + 仓位 + 置信度
    if verdict == "维持原判":
        action = "✅维持原判"
    elif verdict == "修正参数":
        action = "🔧修正参数"
    elif verdict == "降级执行":
        action = "⚠️降级执行"
    elif verdict == "推翻改为反向操作":
        action = "🔄推翻"
    elif verdict == "推翻改为观望":
        action = "🔄推翻→观望"
    else:
        action = "⚪观望"

    meta = (f"{action} · {dir_icon}{dir_text} · {size_map.get(pos_size, '？')} · {conf_map.get(confidence, '？')}\n"
            f"📋交易委员会 ⏱ {_now()}")

    # 执行指令
    entry_low = strategy.get("entry_price_low", 0)
    entry_high = strategy.get("entry_price_high", 0)
    stop_loss = strategy.get("stop_loss", 0)
    take_profit = strategy.get("take_profit", 0)
    exec_plan = strategy.get("execution_plan", "无")
    current = data.get("mark_price", 0) if judge_result else strategy.get("mark_price", 0)  # 尽量获取现价

    # 执行块
    if direction == "neutral":
        exec_block = "🎯 执行指令：当前无操作（观望）"
    else:
        exec_block = (
            f"🎯 执行指令：\n"
            f"方向：{dir_icon}{dir_text}\n"
            f"现价：{current:.0f}\n"
            f"入场：{entry_low:.0f}-{entry_high:.0f} （基于关键支撑/清算区）\n"
            f"止损：{stop_loss:.0f} （跌破关键位）\n"
            f"止盈：{take_profit:.0f} （目标流动性区域）\n"
            f"说明：{_safe(exec_plan)}"
        )

    # 核心逻辑（从 _judge_reasoning 或 reasoning_block_raw 中提取一句话）
    core_logic = ""
    reasoning_block = strategy.get("_reasoning_block_raw", "")
    if reasoning_block:
        # 尝试提取最后一段“核心逻辑”或两行文本
        lines = reasoning_block.split('\n')
        for i, line in enumerate(lines):
            if "核心逻辑" in line:
                core_logic = line.strip()
                break
        if not core_logic and lines:
            core_logic = lines[-1] if len(lines[-1]) > 20 else lines[-2]
        core_logic = _safe(core_logic[:200])
    if not core_logic:
        core_logic = strategy.get("_judge_reasoning", "")
        core_logic = _safe(core_logic[:200])
    if not core_logic:
        core_logic = "无详细理由"

    risk = _safe(strategy.get("risk_note", "请严格设置止损"))

    msg = (f"{title}\n{meta}\n\n"
           f"{exec_block}\n\n"
           f"📋 核心逻辑：{core_logic}\n\n"
           f"⚠️ 风险说明：{risk}")
    return msg


# ========== 兼容旧接口 ==========
def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict) -> str:
    # 直接调用新最终裁决模板
    return format_final_decision(symbol, strategy, judge_result)


# 移除旧版 format_reasoning 等不需要的函数，避免命名空间污染
# （可选：保留 format_reasoning 以备其他地方使用，如果确定无引用可删除）
