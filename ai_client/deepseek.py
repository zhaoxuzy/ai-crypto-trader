import os
import json
import time
import re
from datetime import datetime
from openai import OpenAI
from utils.logger import logger

TICK_SIZE = 0.1
MAX_RETRIES = 2
RETRY_BASE_WAIT = 2
TIMEOUT_SECONDS = 180

def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    # ... 此函数保持您现有的九步 Prompt 完整内容不变 ...
    # 由于内容过长，此处省略，请直接使用您项目中的最新 build_prompt 函数
    pass

def _log_response(prompt: str, content: str, reasoning: str = None):
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/deepseek_{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "content": content, "reasoning": reasoning}, f, ensure_ascii=False, indent=2)
    except:
        pass

def extract_json(content: str) -> str:
    m = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\s*([\s\S]*?)\s*```', content)
    if m:
        return m.group(1).strip()
    start = content.find('{')
    if start == -1:
        raise ValueError("未找到 JSON")
    count = 0
    for i, c in enumerate(content[start:], start):
        if c == '{':
            count += 1
        elif c == '}':
            count -= 1
            if count == 0:
                return content[start:i+1].strip()
    raise ValueError("JSON 未闭合")

def call_deepseek(prompt: str, max_retries: int = MAX_RETRIES) -> dict:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=TIMEOUT_SECONDS
    )
    for attempt in range(max_retries):
        try:
            logger.info(f"DeepSeek 调用 (尝试 {attempt+1}/{max_retries})")
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32768,
                timeout=TIMEOUT_SECONDS
            )
            logger.info(f"实际调用的模型: {resp.model}")
            content = resp.choices[0].message.content or ""
            reasoning = getattr(resp.choices[0].message, 'reasoning_content', None)
            _log_response(prompt, content, reasoning)

            final_content = content.strip() if content else (reasoning or "")
            if not final_content:
                raise ValueError("响应内容为空")

            json_str = extract_json(final_content)
            s = json.loads(json_str)

            s.setdefault("position_size", "none")
            s.setdefault("execution_plan", "")
            s.setdefault("reasoning", "")
            s.setdefault("risk_note", "")
            return s

        except Exception as e:
            logger.warning(f"调用失败: {e}")
            if attempt < max_retries - 1:
                wait_time = RETRY_BASE_WAIT ** (attempt + 1)
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise
    return {}

def round_to_tick(price: float) -> float:
    return round(price / TICK_SIZE) * TICK_SIZE

def _force_neutral(s: dict, reason: str):
    old_risk = s.get('risk_note', '')
    s["direction"] = "neutral"
    s["confidence"] = "low"
    s["position_size"] = "none"
    s["entry_price_low"] = 0
    s["entry_price_high"] = 0
    s["stop_loss"] = 0
    s["take_profit"] = 0
    s["execution_plan"] = ""
    s["reasoning"] += f"\n\n[系统自动干预] 原始信号因校验规则被强制改为观望。原因：{reason}"
    s["risk_note"] = f"[系统干预] 原始方向因校验规则被强制改为观望。" + (f" 原始风险说明：{old_risk}" if old_risk else "")

def validate_strategy(s: dict, data: dict = None) -> tuple[bool, str]:
    direction = s.get("direction")
    if direction not in ["long", "short", "neutral"]:
        return False, f"无效方向: {direction}"

    # 核心数据缺失强制拦截
    if data:
        atr_15m = data.get("atr_15m", 0)
        above_liq = data.get("above_liq", 0)
        below_liq = data.get("below_liq", 0)
        cvd_slope = data.get("cvd_slope", None)

        if above_liq <= 0 and below_liq <= 0 and direction != "neutral":
            _force_neutral(s, "清算数据缺失，强制输出 neutral")
            return True, "已自动修正为观望"

        if atr_15m <= 0 and s.get("confidence") == "high":
            s["confidence"] = "medium"
            logger.warning("核心数据缺失(atr_15m)，置信度强制降级为 medium")
        if cvd_slope is None and s.get("confidence") == "high":
            s["confidence"] = "medium"
            logger.warning("核心数据缺失(cvd_slope)，置信度强制降级为 medium")

    # neutral 信号校验
    if direction == "neutral":
        for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
            if s.get(f, 0) != 0:
                return False, f"neutral 信号不应有非零的 {f}"
        return True, ""

    # 价格字段有效性
    for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
        val = s.get(f)
        if val is None or float(val) <= 0:
            return False, f"缺少或无效的 {f}"

    entry_low = float(s["entry_price_low"])
    entry_high = float(s["entry_price_high"])
    stop_loss = float(s["stop_loss"])
    take_profit = float(s["take_profit"])

    if entry_low > entry_high:
        return False, "入场区间下限大于上限"

    # 止损位几何合理性检查（仅提示，不拦截）
    if direction == "long" and stop_loss >= entry_low:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间下方，请人工确认。"
    elif direction == "short" and stop_loss <= entry_high:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间上方，请人工确认。"

    # 计算盈亏比
    try:
        if direction == "long":
            worst_entry = entry_high
            risk = worst_entry - stop_loss
            reward = take_profit - worst_entry
        else:
            worst_entry = entry_low
            risk = stop_loss - worst_entry
            reward = worst_entry - take_profit

        if risk > 0:
            s["_calculated_rr"] = round(reward / risk, 2)
        else:
            s["_calculated_rr"] = 0.0
    except Exception as e:
        logger.warning(f"计算盈亏比时出错: {e}")
        s["_calculated_rr"] = None

    # decision_summary 与 direction 的一致性校验（强制转为 neutral）
    summary = s.get("decision_summary", {})
    if summary:
        final_dir = summary.get("final_direction", "")
        if final_dir == "看涨" and direction != "long":
            _force_neutral(s, f"decision_summary最终方向为看涨，但direction为{direction}")
            return True, "已自动修正为观望"
        if final_dir == "看跌" and direction != "short":
            _force_neutral(s, f"decision_summary最终方向为看跌，但direction为{direction}")
            return True, "已自动修正为观望"
        if final_dir == "观望" and direction != "neutral":
            _force_neutral(s, f"decision_summary最终方向为观望，但direction为{direction}")
            return True, "已自动修正为观望"

    return True, ""

# ==================== 异议审查官模块 ====================

def build_review_prompt(original_strategy: dict, data: dict, symbol: str, cross_data: dict = None) -> str:
    """构建异议审查官的专用 Prompt"""
    return f"""你是一位顶级加密货币交易策略的异议审查官。你的使命不是自己给出新方向，而是批判性地审查以下策略的每一个环节，找出逻辑漏洞、数据曲解和思考盲点。

【交易标的】{symbol}
【市场数据】（这是策略制定时所依据的全部数据，审查时请严格对照）
{json.dumps(data, ensure_ascii=False, indent=2)}

【原策略裁决】
方向：{original_strategy.get('direction')}
置信度：{original_strategy.get('confidence')}
仓位：{original_strategy.get('position_size')}
入场：{original_strategy.get('entry_price_low')} - {original_strategy.get('entry_price_high')}
止损：{original_strategy.get('stop_loss')}
止盈：{original_strategy.get('take_profit')}

【原策略完整推演过程】（必须与上方市场数据逐一核对）
{original_strategy.get('reasoning', '无推演过程')}

【审查要求】
你必须逐条完成以下审查任务，不得遗漏：

1. **数据真实性审查**：原策略推理中引用的数值是否与上方市场数据一致？是否有夸大、缩小或编造的数据点？列出所有发现的差异，即使只有轻微的不一致。

2. **逻辑漏洞审查**：原策略的推理链条是否存在以下问题？
   - 因果倒置（把结果当原因）
   - 选择性忽略（无视了某个重要的反向数据）
   - 循环论证（用自身结论证明自身）
   列出具体的问题点和对应的推理片段。

3. **仓位与风控合理性审查**：
   - 止损位是否在关键结构位外侧？是否满足1.2倍ATR的硬约束？
   - 仓位大小是否与置信度合理匹配？（高置信度可以重仓，低置信度必须轻仓，两者错配需指出）

4. **方向一致性审查**：
   - 原策略的入场方向是否与推演中的“第一段显著运动”同向？
   - 如果原策略推演为“先跌后涨”却输出做多，必须明确指出。
   - 如果原策略在第七步的权重分配与自身逻辑矛盾，必须指出。

5. **最终判决**：
   基于以上审查，你只能选择以下三种判决之一：
   - **维持原判**：原策略无明显瑕疵，维持原方向和所有参数。
   - **推翻原判，改为观望**：原策略存在严重缺陷（如数据造假、逻辑矛盾、仓位错配），输出 neutral。
   - **修正原判**：原策略方向正确，但参数需要调整（例如止损过近或过远），给出修正后的入场/止损/止盈。

输出JSON（不要代码块）：
{{
  "direction": "long/short/neutral",
  "confidence": "high/medium/low",
  "position_size": "light/medium/heavy/none",
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "execution_plan": "一句话指令",
  "reasoning": "你的完整审查过程和最终判决理由（必须详细写明审查的每一步发现）",
  "risk_note": "修正后的风险说明（如有）",
  "review_verdict": "维持原判/推翻原判改为观望/修正原判",
  "review_issues": ["发现的具体问题1", "问题2", ...]
}}
"""

def call_devils_advocate(original_strategy: dict, data: dict, symbol: str, cross_data: dict = None) -> dict:
    """异议审查官：对原策略进行独立质检，返回最终裁决"""
    review_prompt = build_review_prompt(original_strategy, data, symbol, cross_data)

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=TIMEOUT_SECONDS
    )

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"异议审查官 调用 (尝试 {attempt+1}/{MAX_RETRIES})")
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": review_prompt}],
                max_tokens=8192,
                timeout=TIMEOUT_SECONDS
            )
            content = resp.choices[0].message.content or ""
            _log_response(review_prompt, content)

            if not content.strip():
                raise ValueError("审查官响应为空")

            json_str = extract_json(content)
            reviewed = json.loads(json_str)

            if reviewed.get("direction") not in ["long", "short", "neutral"]:
                raise ValueError("审查官返回无效方向")

            reviewed.setdefault("position_size", original_strategy.get("position_size", "none"))
            reviewed.setdefault("execution_plan", "")
            reviewed.setdefault("risk_note", "")
            reviewed.setdefault("reasoning", "")

            # 记录审查元数据
            reviewed["_reviewed"] = True
            reviewed["_original_direction"] = original_strategy.get("direction")
            reviewed["_review_verdict"] = reviewed.get("review_verdict", "维持原判")
            reviewed["_review_issues"] = reviewed.get("review_issues", [])

            # 如果审查官返回 neutral，强制清空价格数据
            if reviewed["direction"] == "neutral":
                reviewed["entry_price_low"] = 0
                reviewed["entry_price_high"] = 0
                reviewed["stop_loss"] = 0
                reviewed["take_profit"] = 0
                reviewed["position_size"] = "none"

            return reviewed

        except Exception as e:
            logger.warning(f"异议审查官调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                # 失败回退：保留原策略，标记未审查
                logger.warning("审查官调用全部失败，回退到原策略")
                original_strategy["_reviewed"] = False
                return original_strategy

    original_strategy["_reviewed"] = False
    return original_strategy
