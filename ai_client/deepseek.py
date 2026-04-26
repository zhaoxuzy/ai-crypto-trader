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

    if direction == "neutral":
        for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
            if s.get(f, 0) != 0:
                return False, f"neutral 信号不应有非零的 {f}"
        return True, ""

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

    if direction == "long" and stop_loss >= entry_low:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间下方，请人工确认。"
    elif direction == "short" and stop_loss <= entry_high:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间上方，请人工确认。"

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


# ==================== 审查官 B 模块 ====================

def build_reviewer_prompt(original_strategy: dict, data: dict, symbol: str) -> str:
    return f"""你是一位顶级加密货币交易策略的审查官。你的使命是对以下策略进行逐项审查，找出逻辑漏洞、数据曲解和思考盲点。你不给出最终方向，只输出结构化的审查报告。

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
你必须对策略的每一步进行审查，使用“通过/存疑/驳回”三级标注，并附上严重性权重（轻度/中度/重大）和证据链接。

输出JSON（不要代码块）：
{{
  "reviewer_B": {{
    "step_by_step": [
      {{
        "step": "第一步：环境定调",
        "verdict": "通过/存疑/驳回",
        "severity": "轻度/中度/重大",
        "issue": "问题描述（若通过则留空）",
        "evidence": "证据（引用原推理或数据）",
        "suggestion": "修正建议"
      }},
      ... 共九步
    ],
    "summary_severity": "轻度/中度/重大",
    "overall_issues": ["问题1", "问题2", ...]
  }}
}}
"""

def call_reviewer(original_strategy: dict, data: dict, symbol: str) -> dict:
    prompt = build_reviewer_prompt(original_strategy, data, symbol)
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=TIMEOUT_SECONDS
    )
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"审查官B 调用 (尝试 {attempt+1}/{MAX_RETRIES})")
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                timeout=TIMEOUT_SECONDS
            )
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip():
                raise ValueError("审查官响应为空")
            json_str = extract_json(content)
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"审查官B调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                raise
    return {}


# ==================== 终审法官 C 模块 ====================

def build_judge_prompt(original_strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> str:
    return f"""你是一位顶级加密货币交易策略的终审法官。你将收到交易员A的策略和审查官B的审查报告。你的使命是做出最终裁决。

【交易标的】{symbol}
【市场数据】
{json.dumps(data, ensure_ascii=False, indent=2)}

【交易员A的策略】
{json.dumps(original_strategy, ensure_ascii=False, indent=2)}

【审查官B的审查报告】
{json.dumps(reviewer_report, ensure_ascii=False, indent=2)}

【裁决流程】
你必须严格按照以下流程图进行裁决，不得跳过任何步骤：

1. 检查审查报告中是否有“驳回”项。若无驳回 → 维持原判（A级）。
2. 若有驳回，检查A的推演中是否已提前回应并有效驳斥了该驳回点。若已有效回应 → 该驳回无效。
3. 若存在有效驳回，判断该问题是否可以通过修正参数（入场、止损、止盈）解决。如果可以 → 修正参数（B级）。
4. 若问题无法通过修正参数解决，判断问题的严重程度。
   - 轻度/中度 → 降级执行（C级），将原仓位降一级（heavy→medium→light）。
   - 重大 → 推翻改为观望（E级）。

【例外条款】
如果你认为严格按照上述流程图会导致明显违背市场逻辑的结果，你可以绕过流程，但必须同时满足以下三个条件：
1. 在裁决书中显式声明：“[例外触发] 我选择不遵循标准流程图，因为……（列出具体原因，必须引用数据）”；
2. 扮演一个与你最终裁决方向相反的交易员，用流程图规定的方式攻击你的决定，并展示你如何驳斥它；
3. 再次确认：最终裁决是否仍然比按流程图执行更优？

输出JSON（不要代码块）：
{{
  "judge_C": {{
    "final_verdict": "维持原判/修正参数/降级执行/补充条件执行/推翻改为观望",
    "verdict_level": "A/B/C/D/E",
    "exception_used": "是/否",
    "exception_reason": "若使用例外条款，填写原因",
    "final_direction": "long/short/neutral",
    "final_confidence": "high/medium/low",
    "final_position_size": "light/medium/heavy/none",
    "entry_price_low": 0.0,
    "entry_price_high": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "execution_plan": "一句话指令",
    "reasoning": "你的完整裁决过程，包括对每个驳回项的回应",
    "risk_note": "风险说明"
  }}
}}
"""

def call_judge(original_strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    prompt = build_judge_prompt(original_strategy, reviewer_report, data, symbol)
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=TIMEOUT_SECONDS
    )
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"终审法官C 调用 (尝试 {attempt+1}/{MAX_RETRIES})")
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                timeout=TIMEOUT_SECONDS
            )
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip():
                raise ValueError("法官响应为空")
            json_str = extract_json(content)
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"法官C调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                raise
    return {}


def apply_final_verdict(original_strategy: dict, judge_result: dict) -> dict:
    """将法官C的最终判决应用到策略上"""
    verdict = judge_result.get("judge_C", {}).get("final_verdict", "维持原判")
    final = judge_result.get("judge_C", {})

    # 保留原策略的元数据
    original_strategy["_reviewed"] = True
    original_strategy["_original_direction"] = original_strategy.get("direction")
    original_strategy["_review_verdict"] = verdict
    original_strategy["_review_issues"] = final.get("overall_issues", [])

    if verdict == "维持原判":
        return original_strategy

    elif verdict == "修正参数":
        original_strategy["entry_price_low"] = final.get("entry_price_low", original_strategy["entry_price_low"])
        original_strategy["entry_price_high"] = final.get("entry_price_high", original_strategy["entry_price_high"])
        original_strategy["stop_loss"] = final.get("stop_loss", original_strategy["stop_loss"])
        original_strategy["take_profit"] = final.get("take_profit", original_strategy["take_profit"])
        return original_strategy

    elif verdict == "降级执行":
        size_map = {"heavy": "medium", "medium": "light", "light": "light"}
        original_strategy["position_size"] = size_map.get(original_strategy.get("position_size", "light"), "light")
        return original_strategy

    elif verdict == "补充条件执行":
        original_strategy["execution_plan"] = final.get("execution_plan", original_strategy.get("execution_plan", ""))
        return original_strategy

    elif verdict == "推翻改为观望":
        _force_neutral(original_strategy, f"法官判决: {verdict}")
        return original_strategy

    return original_strategy
