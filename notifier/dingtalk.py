import re, json
from datetime import datetime
from utils.logger import logger

def send_dingtalk_message(msg: str, title: str = ""):
    """发送钉钉消息，需实现webhook调用（已存在的实现不修改）"""
    # 此函数通常已在您项目中实现，此处保留占位
    pass

def format_strategy_message(symbol: str, strategy: dict, data: dict = None) -> str:
    """
    生成交易员策略推送消息（无推演概要版）
    """
    direction = strategy.get("direction", "neutral")
    direction_emoji = {"long": "🟢做多", "short": "🔴做空", "neutral": "⚪观望"}.get(direction, "⚪观望")
    confidence = strategy.get("confidence", "?")
    position_size = strategy.get("position_size", "?")
    entry_low = strategy.get("entry_price_low", 0)
    entry_high = strategy.get("entry_price_high", 0)
    stop_loss = strategy.get("stop_loss", 0)
    take_profit = strategy.get("take_profit", 0)
    mark_price = data.get("mark_price", 0) if data else 0
    risk_note = strategy.get("risk_note", "")
    execution_plan = strategy.get("execution_plan", "")
    risk_reward = strategy.get("risk_reward_ratio", 0)
    data_constraints = strategy.get("data_quality_constraints", "")
    emergency = strategy.get("emergency_mode", "否")
    cross_action = strategy.get("cross_coin_action", "")

    msg = f"""### 策略｜{symbol} 初步方案 ⏳ {datetime.now().strftime('%m-%d %H:%M')}

{direction_emoji} | 现价 {mark_price:.2f} | 入场 {entry_low:.2f}-{entry_high:.2f} | 止损 {stop_loss:.2f} | 止盈 {take_profit:.2f}
置信度 {'⭐'*3 if confidence=='high' else '⭐⭐' if confidence=='medium' else '⭐'} | 仓位 {'💰💰💰重仓' if position_size=='heavy' else '💰💰中仓' if position_size=='medium' else '💰轻仓' if position_size=='light' else '🚫无'}

> 📊 **核心摘要**：{data_constraints if data_constraints else '无特殊数据质量约束'} | 紧急模式：{emergency} | 跨币种：{cross_action}

"""
    if risk_note:
        msg += f"\n---\n### ❗️ 风险提示\n{risk_note}\n"
    if execution_plan:
        msg += f"\n---\n### ⚡ 执行\n{execution_plan}\n"
    if risk_reward > 0:
        msg += f"\n**盈亏比**：{risk_reward:.2f}"

    return msg

def format_review_message(symbol: str, strategy: dict, reviewer_report: dict, data: dict = None) -> str:
    """
    生成审计官审查报告消息
    """
    verdict = reviewer_report.get("overall_verdict", "未知")
    max_severity = reviewer_report.get("max_severity", "无")
    severity_summary = reviewer_report.get("severity_summary", {})
    full_report = reviewer_report.get("full_report", "无")
    
    sev_text = f"严重：{severity_summary.get('严重',0)} 中等：{severity_summary.get('中等',0)} 轻度：{severity_summary.get('轻度',0)}"
    direction = strategy.get("direction", "?")
    position_size = strategy.get("position_size", "?")
    
    msg = f"""### 策略｜{symbol} 审计报告 📋 {datetime.now().strftime('%m-%d %H:%M')}
{verdict} | 严重性：{max_severity} | {sev_text}
原方向：{direction} | 原仓位：{position_size}

{full_report}
"""
    return msg

def format_judge_message(symbol: str, strategy: dict, judge_result: dict, data: dict = None) -> str:
    """
    生成交易委员会裁决消息
    """
    verdict = judge_result.get("final_verdict", "未知")
    final_direction = judge_result.get("final_direction", "neutral")
    final_confidence = judge_result.get("final_confidence", "?")
    final_position = judge_result.get("final_position_size", "?")
    entry_low = judge_result.get("entry_price_low", 0)
    entry_high = judge_result.get("entry_price_high", 0)
    stop_loss = judge_result.get("stop_loss", 0)
    take_profit = judge_result.get("take_profit", 0)
    reasoning = judge_result.get("final_reasoning", "")
    weighted_signal = judge_result.get("weighted_signal", {})
    
    msg = f"""### 策略｜{symbol} 最终裁决 ⚖️ {datetime.now().strftime('%m-%d %H:%M')}

{verdict} | {"🟢做多" if final_direction=="long" else "🔴做空" if final_direction=="short" else "⚪观望"} | 置信度 {'⭐'*3 if final_confidence=='high' else '⭐⭐' if final_confidence=='medium' else '⭐'} | 仓位 {'💰💰💰重仓' if final_position=='heavy' else '💰💰中仓' if final_position=='medium' else '💰轻仓' if final_position=='light' else '🚫无'}
"""
    if reasoning:
        msg += f"\n---\n### 📜 裁决理由\n{reasoning}\n"
    if weighted_signal:
        weights_str = " | ".join([f"{k}:{v}" for k,v in weighted_signal.items()])
        msg += f"\n---\n### ⚖️ 加权信号\n{weights_str}\n"
    return msg

def format_final_decision(symbol: str, strategy: dict, judge_result: dict, data: dict = None) -> str:
    """
    生成最终策略推送消息（汇总版）
    """
    return format_judge_message(symbol, strategy, judge_result, data)
