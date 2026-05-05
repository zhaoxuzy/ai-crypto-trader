"""
notifier/dingtalk.py — 钉钉消息推送模块 (修复版)
修复点：
1. format_review_message 中 severity_counts 直接从 reviewer_report 获取
2. format_final_decision 中展示完整裁决书正文
"""
import json
import requests
from datetime import datetime
from utils.logger import logger

def _send_dingtalk(webhook_url: str, payload: dict, secret: str = None) -> bool:
    """发送钉钉消息，成功返回 True"""
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get("errcode") == 0:
                return True
            else:
                logger.error(f"钉钉返回错误: {result}")
                return False
        else:
            logger.error(f"钉钉请求失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"钉钉推送异常: {e}")
        return False

def send_dingtalk_message(text: str, title: str = "交易策略通知") -> bool:
    """发送 Markdown 消息到钉钉"""
    webhook_url = os.getenv("DINGTALK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("未配置 DINGTALK_WEBHOOK_URL，跳过推送")
        return False
    payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": title[:30],  # 钉钉标题限制
            "text": text
        }
    }
    return _send_dingtalk(webhook_url, payload)

def format_strategy_message(symbol: str, strategy: dict, data: dict = None) -> str:
    """交易员初步策略消息格式"""
    direction = strategy.get("direction", "neutral")
    confidence = strategy.get("confidence", "中")
    position = strategy.get("position_size", "无")
    entry_low = strategy.get("entry_price_low", 0) or 0
    entry_high = strategy.get("entry_price_high", 0) or 0
    stop = strategy.get("stop_loss", 0) or 0
    profit = strategy.get("take_profit", 0) or 0
    reasoning = strategy.get("reasoning", "")

    dir_map = {"long": "做多", "short": "做空", "neutral": "观望"}
    pos_map = {"heavy": "重仓", "medium": "中仓", "light": "轻仓", "none": "无"}
    conf_map = {"high": "高", "medium": "中", "low": "低"}

    msg = f"### 策略｜{symbol} 🧠 建议 {datetime.now().strftime('%m-%d %H:%M')}\n"
    msg += f"{dir_map.get(direction, '观望')} | 现价 {data.get('mark_price', 0) if data else 0:.0f} | 入场 {entry_low}-{entry_high} | 止损 {stop} | 止盈 {profit} | 置信度 {conf_map.get(confidence, '?')} | 仓位 {pos_map.get(position, '?')}\n\n"
    msg += f"**推演过程**\n```\n{reasoning[:2000]}\n```"
    return msg

def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict = None) -> str:
    """风控审计报告消息格式（修复版）"""
    direction_map = {"long": "做多", "short": "做空", "neutral": "观望"}
    pos_map = {"heavy": "重仓", "medium": "中仓", "light": "轻仓", "none": "无"}
    
    direction = direction_map.get(strategy.get("direction", "neutral"), "观望")
    position = pos_map.get(strategy.get("position_size", "none"), "无")
    
    # 直接使用审计报告里的 severity_counts，而不是自己解析文本
    severity = reviewer_report.get("severity_counts", {"严重": 0, "中等": 0, "轻度": 0})
    severe = severity.get("严重", 0)
    medium = severity.get("中等", 0)
    low = severity.get("轻度", 0)
    
    verdict = reviewer_report.get("verdict", "通过")
    max_sev = reviewer_report.get("max_severity", "无")
    full_report = reviewer_report.get("full_report", "")
    
    title = f"⚡ 审计 {verdict}"
    msg = f"### 策略｜{symbol} ⚡ 审计 {verdict} {datetime.now().strftime('%m-%d %H:%M')}\n"
    msg += f"严重 {severe} | 中等 {medium} | 轻微 {low}\n\n"
    msg += f"**方向**: {direction} | **仓位**: {position} | **置信度**: {strategy.get('confidence', '?')}\n\n"
    msg += f"**审计报告**\n{full_report[:2000]}"
    
    return msg

def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict = None) -> str:
    """交易委员会裁决消息格式（已废弃，保留兼容）"""
    return format_final_decision(symbol, strategy, judge_result, data)

def format_final_decision(symbol: str, strategy: dict, judge_result: dict, data: dict = None) -> str:
    """最终裁决消息格式（修复版，展示完整裁决书）"""
    verdict = judge_result.get("final_verdict", "维持原判")
    direction = judge_result.get("final_direction", strategy.get("direction", "neutral"))
    confidence = judge_result.get("final_confidence", strategy.get("confidence", "中"))
    position = judge_result.get("final_position_size", strategy.get("position_size", "无"))
    entry_low = judge_result.get("entry_price_low", 0) or 0
    entry_high = judge_result.get("entry_price_high", 0) or 0
    stop = judge_result.get("stop_loss", 0) or 0
    profit = judge_result.get("take_profit", 0) or 0

    # 方向/仓位映射
    dir_display = {"long": "做多", "short": "做空", "neutral": "观望"}.get(direction, "观望")
    pos_display = {"heavy": "重仓", "medium": "中仓", "light": "轻仓", "none": "无"}.get(position, "无")
    conf_display = {"high": "高", "medium": "中", "low": "低"}.get(confidence, "中")

    # 构建标题行
    symbol_display = "🔴做空" if direction == "short" else ("🟢做多" if direction == "long" else "⚪观望")
    msg = f"### 策略｜{symbol} 📋 裁决 {'✅维持' if verdict=='维持原判' else '🔄推翻'} {datetime.now().strftime('%m-%d %H:%M')}\n"
    msg += f"{symbol_display} | 现价 {data.get('mark_price',0) if data else 0:.0f} | 入场 {entry_low}-{entry_high} | 止损 {stop} | 止盈 {profit} | 置信度 {conf_display} | 仓位 {pos_display}\n\n"

    # 裁决书正文（完整展示）
    final_reasoning = judge_result.get("final_reasoning", "") or "无裁决内容"
    msg += f"**裁决内容**\n```\n{final_reasoning}\n```"

    # 审计采纳标记
    if judge_result.get("audit_adopted", False):
        msg += f"\n> 审计意见已采纳"
    return msg