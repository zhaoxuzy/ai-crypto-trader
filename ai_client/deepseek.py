"""
deepseek.py — 五因子方向决策矩阵版
核心改进：
1. 用五因子量化打分替代模糊的方向判断
2. 强制总分在 -1~+1 之间时输出观望
3. 系统锚点 direction_bias 作为最终校验
4. 保留所有数据字段供入场计划使用
"""

import os
import json
import time
import re
from datetime import datetime
from openai import OpenAI
from utils.logger import logger

TICK_SIZE = 0.1
MAX_RETRIES = 3
RETRY_BASE_WAIT = 2
TIMEOUT_SECONDS = 180

FAST_MODEL = "deepseek-v4-pro"
REASONING_MODEL = "deepseek-v4-pro"


# ---------- 辅助函数 ----------
def _safe_float(pattern, text, default=0.0):
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return default
    return default


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


def round_to_tick(price: float) -> float:
    return round(price / TICK_SIZE) * TICK_SIZE


def _force_neutral(s: dict, reason: str):
    s["direction"] = "neutral"
    s["confidence"] = "low"
    s["position_size"] = "none"
    s["entry_price_low"] = 0
    s["entry_price_high"] = 0
    s["stop_loss"] = 0
    s["take_profit"] = 0
    s["execution_plan"] = ""
    s["reasoning"] = (s.get("reasoning", "") + f"\n\n[系统强制观望，原因：{reason}]").strip()
    s["risk_note"] = f"观望。{reason}"


def validate_strategy(s: dict, data: dict = None) -> tuple[bool, str]:
    direction = s.get("direction")
    if direction not in ["long", "short", "neutral"]:
        return False, f"无效方向: {direction}"

    if data:
        atr_15m = data.get("atr_15m", 0)
        above_liq = data.get("above_liq", 0)
        below_liq = data.get("below_liq", 0)
        mark_price = data.get("mark_price", 0)
        direction_bias = data.get("direction_bias", 0.0)

        if (not above_liq or above_liq <= 0) and (not below_liq or below_liq <= 0) and direction != "neutral":
            _force_neutral(s, "清算数据缺失，强制 neutral")
            return True, "已自动修正为观望"
        if atr_15m <= 0 or mark_price <= 0:
            if direction != "neutral":
                _force_neutral(s, "ATR 或价格缺失，强制 neutral")
                return True, "已自动修正为观望"

        if atr_15m <= 0 and s.get("confidence") == "high":
            s["confidence"] = "medium"
            logger.warning("核心数据缺失(atr_15m)，置信度强制降级为 medium")
        if data.get("cvd_slope") is None and s.get("confidence") == "high":
            s["confidence"] = "medium"
            logger.warning("核心数据缺失(cvd_slope)，置信度强制降级为 medium")

        if abs(direction_bias) > 0.4 and direction != "neutral":
            if (direction_bias > 0 and direction == "short") or (direction_bias < 0 and direction == "long"):
                if s.get("confidence") == "high":
                    s["confidence"] = "medium"
                    logger.warning(f"方向与 direction_bias({direction_bias:.3f})冲突，置信度降级为 medium")

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
        logger.warning(f"入场区间下限({entry_low})大于上限({entry_high})，已自动交换")
        s["entry_price_low"], s["entry_price_high"] = entry_high, entry_low

    if direction == "long" and stop_loss >= entry_low:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间下方。"
    elif direction == "short" and stop_loss <= entry_high:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间上方。"

    try:
        if direction == "long":
            risk = entry_high - stop_loss
            reward = take_profit - entry_high
        else:
            risk = stop_loss - entry_low
            reward = entry_low - take_profit
        if risk > 0:
            s["_calculated_rr"] = round(reward / risk, 2)
        else:
            s["_calculated_rr"] = 0.0
    except:
        s["_calculated_rr"] = None

    return True, ""


# ------------------- 构建策略提示词 -------------------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    if cross_symbol is None:
        cross_symbol = "ETH" if symbol == "BTC" else "BTC"

    # ---------- 安全取值 ----------
    def safe_val(key, default=0.0, scale=1.0, fmt=".2f"):
        raw = data.get(key)
        if raw is None:
            return ("缺失", True)
        try:
            val = float(raw) * scale
            return (f"{val:{fmt}}", False)
        except (ValueError, TypeError):
            return ("缺失", True)

    # ---------- 提取所有字段 ----------
    mark_price_str, _ = safe_val('mark_price', fmt=".2f")
    price_percentile_str, _ = safe_val('price_percentile', 50.0, fmt=".0f")
    atr_str, _ = safe_val('atr', fmt=".2f")
    atr_1h_ratio_str, _ = safe_val('atr_1h_ratio', fmt=".1f")
    vol_factor_str, _ = safe_val('vol_factor', 1.0, fmt=".2f")
    cgdi_str, _ = safe_val('cgdi_current', fmt=".0f")
    cgdi_perc_str, _ = safe_val('cgdi_percentile', 50.0, fmt=".0f")
    fear_greed = data.get('fear_greed', 50)
    fear_greed_prev = data.get('fear_greed_prev_7d', 50)

    above_liq_str, _ = safe_val('above_liq', scale=1/1e9, fmt=".2f")
    below_liq_str, _ = safe_val('below_liq', scale=1/1e9, fmt=".2f")
    liq_ratio_str, _ = safe_val('liq_ratio', fmt=".2f")
    above_cluster = data.get('above_cluster', 'N/A')
    above_trigger = data.get('above_trigger', 'N/A')
    below_cluster = data.get('below_cluster', 'N/A')
    below_trigger = data.get('below_trigger', 'N/A')

    large_sell_str, _ = safe_val('large_sell_value', scale=1/1e6, fmt=".1f")
    large_buy_str, _ = safe_val('large_buy_value', scale=1/1e6, fmt=".1f")
    pressure_str, _ = safe_val('large_order_pressure', fmt=".3f")
    ob_imbalance_str, _ = safe_val('orderbook_imbalance', fmt=".3f")
    lure_str, _ = safe_val('lure_risk_factor', fmt=".2f")

    oi_chg_str, _ = safe_val('oi_change_24h', fmt="+.1f")
    agg_oi_chg_str, _ = safe_val('agg_oi_change_24h', fmt="+.1f")
    fund_rate_str, _ = safe_val('funding_rate', fmt=".4f")
    fund_perc_str, _ = safe_val('funding_percentile', 50.0, fmt=".0f")
    top_ls_perc_str, _ = safe_val('top_ls_percentile', 50.0, fmt=".0f")

    lth_rp_str, _ = safe_val('lth_realized_price', fmt=".2f")
    sth_rp_str, _ = safe_val('sth_realized_price', fmt=".2f")
    sth_sopr_str, _ = safe_val('sth_sopr', 1.0, fmt=".3f")
    stable_trend_str, _ = safe_val('stablecoin_trend_7d', fmt="+.1f")
    btc_dom_trend_str, _ = safe_val('btc_dominance_trend_7d', fmt="+.1f")
    borrow_str, _ = safe_val('borrow_rate', scale=100, fmt=".2f")

    max_pain_str, _ = safe_val('max_pain', fmt=".2f")
    pc_str, _ = safe_val('put_call_ratio', fmt=".4f")

    cvd_slope_str, _ = safe_val('cvd_slope', fmt=".4f")
    taker_str, _ = safe_val('taker_ratio_1h', fmt=".3f")
    nf24h_str, _ = safe_val('netflow_24h', scale=1/1e6, fmt=".1f")
    spot_24h_str, _ = safe_val('spot_netflow_24h', scale=1/1e6, fmt=".1f")
    basis_perc_str, _ = safe_val('basis_percentile', 50.0, fmt=".0f")

    direction_bias = data.get('direction_bias', 0.0)

    # 跨币种
    cross_context = ""
    if eth_data:
        cross_oi = eth_data.get('oi_change_24h', 0)
        cross_context = f"【{cross_symbol}关键对比】OI 24h变化：{cross_oi:+.1f}%，顶多空分位：{eth_data.get('top_ls_percentile', 50):.0f}%"

    # ---------- 构建 Prompt ----------
    prompt = f"""你是一位顶尖的加密货币交易员，遵循严谨的量化决策框架。你的任务是筛选出高胜率、高盈亏比的交易机会。

【核心规则】
- **方向决策矩阵**：你必须完成一个五因子的量化打分表，根据最终得分决定方向。
- **矛盾处理**：若总得分在-1到+1之间，必须输出"观望"。
- **锚点服从**：若你的判断与系统锚点(direction_bias)冲突，必须"观望"并解释原因。
- **风险回报比**：只有当潜在盈亏比大于2:1时，才可入场。

【{symbol} 市场核心数据】
现价：{mark_price_str} (7日分位{price_percentile_str}%)
波动性：ATR(4h): {atr_str}，1h波动率：{atr_1h_ratio_str}%，波动因子：{vol_factor_str}
情绪：恐慌贪婪{fear_greed}， CGDI：{cgdi_str} (分位{cgdi_perc_str}%)
基差分位：{basis_perc_str}%

【任务一：方向决策矩阵（五因子量化打分）】
请根据以下五个因子打分（+1=多头，-1=空头，0=中性），必须引用具体数据：

| 因子 | 关键数据（填入表格） | 打分 |
|------|---------------------|------|
| 资金流与持仓 | 期货24h净流：{nf24h_str}M，现货24h净流：{spot_24h_str}M，OI 24h变化：{oi_chg_str}% | |
| 价值与成本 | LTH成本：{lth_rp_str}，STH成本：{sth_rp_str}，STH SOPR：{sth_sopr_str} | |
| 情绪极端度 | 恐慌贪婪：{fear_greed}，P/B比：{pc_str}，费率分位：{fund_perc_str}% | |
| 动能与趋势 | CVD斜率：{cvd_slope_str}，主动买卖比：{taker_str}，ATR：{atr_str} | |
| 宏观与链上 | 稳定币趋势：{stable_trend_str}%，BTC.D趋势：{btc_dom_trend_str}%，借贷利率：{borrow_str}% | |

**矩阵总分：__ / 5**
**方向判断**：总分≥2→做多，≤-2→做空，-1~1→观望。

【任务二：入场计划】（仅当任务一方向非观望时填写）
利用以下完整数据制定精确的入场、止损和止盈计划。

清算地形：上方{above_liq_str}B（{above_trigger}点），下方{below_liq_str}B（{below_trigger}点）
期权：最大痛点{max_pain_str}，P/B比{pc_str}
大单挂单：卖{large_sell_str}M / 买{large_buy_str}M，压迫比{pressure_str}
订单簿失衡率：{ob_imbalance_str}，诱饵风险：{lure_str}
全市场OI：{agg_oi_chg_str}%，大户分位：{top_ls_perc_str}%
跨币种：{cross_context}

入场区间：___ - ___ (依据：___)
止损：___ (依据：___)
止盈：___ (依据：___)

【强制输出格式】
{{
  "direction": "做多 / 做空 / 观望",
  "confidence": "高 / 中 / 低",
  "position_size": "重仓 / 中仓 / 轻仓 / 无",
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "execution_plan": "一句话指令",
  "reasoning": "必须包含矩阵打分、方向判断依据(至少三个数据字段)、入场计划。",
  "risk_note": "核心风险（20字内）"
}}
"""
    return prompt


# ------------------- 首席交易员调用 -------------------
def call_deepseek(prompt: str, max_retries: int = MAX_RETRIES) -> dict:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=TIMEOUT_SECONDS
    )
    for attempt in range(max_retries):
        try:
            logger.info(f"首席交易员 调用 (尝试 {attempt+1}/{max_retries}) [模型: {FAST_MODEL}]")
            resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                timeout=TIMEOUT_SECONDS
            )
            logger.info(f"实际调用的模型: {resp.model}")
            content = resp.choices[0].message.content or ""
            reasoning = getattr(resp.choices[0].message, 'reasoning_content', None)
            _log_response(prompt, content, reasoning)

            final_content = content.strip() if content else (reasoning or "")
            if not final_content:
                raise ValueError("响应内容为空")

            try:
                json_str = extract_json(final_content)
                s = json.loads(json_str)
            except Exception as e:
                logger.error(f"JSON 解析失败: {e}\n原始内容: {final_content[:500]}")
                raise ValueError(f"JSON 解析失败: {e}")

            # 方向标准化
            dir_map = {"做多": "long", "做空": "short", "观望": "neutral",
                       "long": "long", "short": "short", "neutral": "neutral"}
            raw_dir = s.get("direction", "")
            if raw_dir in dir_map:
                s["direction"] = dir_map[raw_dir]
            else:
                reasoning_text = s.get("reasoning", "")
                final_decision_match = re.search(r'方向[：:]\s*(做多|做空|观望|long|short|neutral)', reasoning_text)
                if final_decision_match:
                    s["direction"] = dir_map.get(final_decision_match.group(1), "neutral")
                else:
                    s["direction"] = "neutral"

            # 仓位标准化
            pos_map = {"轻仓": "light", "中仓": "medium", "重仓": "heavy", "无": "none",
                       "light": "light", "medium": "medium", "heavy": "heavy", "none": "none"}
            raw_pos = s.get("position_size", "")
            s["position_size"] = pos_map.get(raw_pos, "none")

            # 置信度标准化
            conf_map = {"高": "high", "中": "medium", "低": "low",
                        "high": "high", "medium": "medium", "low": "low"}
            raw_conf = s.get("confidence", "")
            s["confidence"] = conf_map.get(raw_conf, "medium")

            s.setdefault("execution_plan", "")
            s.setdefault("reasoning", "")
            s.setdefault("risk_note", "")
            s["_model_used"] = resp.model
            return s

        except Exception as e:
            logger.warning(f"首席交易员调用失败: {e}")
            if attempt < max_retries - 1:
                wait_time = RETRY_BASE_WAIT ** (attempt + 1)
                time.sleep(wait_time)
            else:
                return {
                    "direction": "neutral", "confidence": "low", "position_size": "none",
                    "entry_price_low": 0, "entry_price_high": 0, "stop_loss": 0, "take_profit": 0,
                    "execution_plan": "模型调用失败", "reasoning": "调用失败", "risk_note": "模型调用失败",
                    "_model_used": "fallback"
                }


# ------------------- 风控审计官 -------------------
def build_reviewer_prompt(original_strategy: dict, data: dict, symbol: str) -> str:
    direction_bias = data.get('direction_bias', 0.0)
    return f"""你是风控审计官。请基于以下数据核查首席交易员的策略。

【交易标的】{symbol}
【系统锚点】direction_bias={direction_bias:.3f}

【交易员策略】
方向：{original_strategy.get('direction')}
入场：{original_strategy.get('entry_price_low')}-{original_strategy.get('entry_price_high')}
止损：{original_strategy.get('stop_loss')}　止盈：{original_strategy.get('take_profit')}

【交易员推演】
{original_strategy.get('reasoning', '无')}

请按五节模板输出审计报告（一、遗漏指标与分析缺失；二、数据与解读错误；三、逻辑错误；四、关键反证提示；五、博弈层面审视）。每条末尾标注[严重性：高/中/低]。只输出报告本身。
"""


def call_reviewer(original_strategy: dict, data: dict, symbol: str) -> dict:
    prompt = build_reviewer_prompt(original_strategy, data, symbol)
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=FAST_MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=4096, timeout=120
            )
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip():
                raise ValueError("审计官响应为空")

            severity_counts = {"高": 0, "中": 0, "低": 0}
            for line in content.split('\n'):
                for level in ["高", "中", "低"]:
                    if f"严重性：{level}" in line:
                        severity_counts[level] += 1

            verdict = "通过"
            if severity_counts["高"] > 0:
                verdict = "驳回"
            elif severity_counts["中"] > 0 or severity_counts["低"] > 0:
                verdict = "存疑"

            return {"verdict": verdict, "full_report": content, "severity_counts": severity_counts, "_model_used": resp.model}
        except Exception as e:
            logger.warning(f"风控审计官调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                return {"verdict": "通过", "full_report": "审计官调用失败", "_model_used": "fallback"}


# ------------------- 交易委员会 -------------------
def build_judge_prompt(original_strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> str:
    direction_bias = data.get('direction_bias', 0.0)
    report = reviewer_report.get('full_report', '无审计报告')
    return f"""你是交易委员会主席。请基于以下数据公正裁决。

【{symbol}】现价：{data.get('mark_price', 0):.2f}
【系统锚点】direction_bias={direction_bias:.3f}

【交易员策略】方向：{original_strategy.get('direction')}
入场：{original_strategy.get('entry_price_low')}-{original_strategy.get('entry_price_high')}
止损：{original_strategy.get('stop_loss')}　止盈：{original_strategy.get('take_profit')}

【审计报告】{report}

请输出：
1. 方向锚点检查
2. 审计指控逐条裁决
3. 最终判决（维持原判/推翻）
4. 最终合约策略（币种、方向、仓位、入场区间、止损、止盈、说明）
"""


def call_judge(original_strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    prompt = build_judge_prompt(original_strategy, reviewer_report, data, symbol)
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=REASONING_MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=16384, timeout=120
            )
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip():
                raise ValueError("交易委员会响应为空")

            # 提取判决
            verdict = "维持原判"
            verdict_match = re.search(r'📌\s*最终判决[：:]\s*(.*)', content)
            if verdict_match:
                clean = verdict_match.group(1).replace('*', '').strip()
                verdict = "维持原判" if "维持" in clean else "推翻"

            if verdict == "维持原判":
                return {
                    "judge_C": {
                        "final_verdict": "维持原判",
                        "final_direction": original_strategy.get("direction", "neutral"),
                        "final_confidence": original_strategy.get("confidence", "medium"),
                        "final_position_size": original_strategy.get("position_size", "none"),
                        "entry_price_low": original_strategy.get("entry_price_low", 0) or 0,
                        "entry_price_high": original_strategy.get("entry_price_high", 0) or 0,
                        "stop_loss": original_strategy.get("stop_loss", 0) or 0,
                        "take_profit": original_strategy.get("take_profit", 0) or 0,
                        "execution_plan": original_strategy.get("execution_plan", ""),
                        "reasoning": content,
                        "risk_note": original_strategy.get("risk_note", ""),
                        "_model_used": resp.model
                    }
                }
            else:
                # 推翻时提取新策略
                direction = "neutral"
                position_size = "none"
                entry_low = entry_high = stop_loss = take_profit = 0.0
                exec_match = re.search(r'🎯.*?方向[：:]\s*([^\s\n]+)', content)
                if exec_match:
                    raw = exec_match.group(1).replace('*', '')
                    dir_map = {"做多": "long", "做空": "short", "观望": "neutral"}
                    direction = dir_map.get(raw, "neutral")
                return {
                    "judge_C": {
                        "final_verdict": "推翻",
                        "final_direction": direction,
                        "final_confidence": "medium",
                        "final_position_size": position_size,
                        "entry_price_low": entry_low,
                        "entry_price_high": entry_high,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "reasoning": content,
                        "risk_note": "",
                        "_model_used": resp.model
                    }
                }
        except Exception as e:
            logger.warning(f"交易委员会调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                return {"judge_C": {"final_verdict": "维持原判", "_model_used": "fallback"}}


def apply_final_verdict(original_strategy: dict, judge_result: dict, reviewer_report: dict = None) -> dict:
    verdict = judge_result.get("judge_C", {}).get("final_verdict", "维持原判")
    final = judge_result.get("judge_C", {})
    logger.info(f"应用最终决议: {verdict}")

    original_strategy["_reviewed"] = True
    original_strategy["_original_direction"] = original_strategy.get("direction")
    original_strategy["_review_verdict"] = verdict

    if verdict == "维持原判":
        if "risk_note" in final:
            original_strategy["risk_note"] = final["risk_note"]
    else:
        new_dir = final.get("final_direction", "neutral")
        if new_dir == "neutral":
            _force_neutral(original_strategy, "交易委员会推翻并改为观望")
        else:
            original_strategy["direction"] = new_dir
            original_strategy["confidence"] = final.get("final_confidence", "medium")
            original_strategy["position_size"] = final.get("final_position_size", "none")
            original_strategy["entry_price_low"] = final.get("entry_price_low", 0) or 0
            original_strategy["entry_price_high"] = final.get("entry_price_high", 0) or 0
            original_strategy["stop_loss"] = final.get("stop_loss", 0) or 0
            original_strategy["take_profit"] = final.get("take_profit", 0) or 0

    if original_strategy.get("direction") == "neutral":
        original_strategy["entry_price_low"] = 0
        original_strategy["entry_price_high"] = 0
        original_strategy["stop_loss"] = 0
        original_strategy["take_profit"] = 0
        original_strategy["position_size"] = "none"

    return original_strategy