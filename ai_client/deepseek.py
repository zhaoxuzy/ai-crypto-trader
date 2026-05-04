"""
deepseek.py — 单Agent深度分析框架
核心流程：交易员接收全部数据 → 深度分析（周期定位 + 三组投票 + 战术定位 + 逻辑终审） → 硬编码校验
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


# ---------- 辅助函数 ----------
def _log_response(prompt: str, content: str, reasoning: str = None):
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/deepseek_{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "content": content, "reasoning": reasoning}, f, ensure_ascii=False, indent=2)
    except:
        pass


def extract_json(content: str) -> str:
    """增强的 JSON 提取器，自动修补未闭合的 JSON"""
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
    last_valid_end = -1
    for i, c in enumerate(content[start:], start):
        if c == '{':
            count += 1
        elif c == '}':
            count -= 1
            if count == 0:
                return content[start:i+1].strip()
            if count < 0:
                break
        if count == 0:
            last_valid_end = i
    if last_valid_end != -1:
        partial = content[start:last_valid_end+1] + '}'
        logger.warning("JSON 未正确闭合，已尝试自动修补")
        return partial
    patched = content[start:] + '}}'
    logger.warning("JSON 严重损坏，尝试暴力修补")
    return patched


def _force_neutral(s: dict, reason: str):
    s["direction"] = "neutral"
    s["confidence"] = "低"
    s["position_size"] = "无"
    s["entry_price_low"] = 0
    s["entry_price_high"] = 0
    s["stop_loss"] = 0
    s["take_profit"] = 0
    s["execution_plan"] = ""
    s["reasoning"] = (s.get("reasoning", "") + f"\n\n[系统强制观望，原因：{reason}]").strip()
    s["risk_note"] = f"观望。{reason}"


def validate_strategy(s: dict, data: dict = None) -> tuple[bool, str]:
    """硬编码校验：致命缺失、锚点冲突、方向合法性"""
    direction = s.get("direction")
    if direction not in ["long", "short", "neutral"]:
        return False, f"无效方向: {direction}"

    if data:
        atr_15m = data.get("atr_15m", 0)
        mark_price = data.get("mark_price", 0)
        above_liq = data.get("above_liq", 0)
        below_liq = data.get("below_liq", 0)
        direction_bias = data.get("direction_bias", 0.0)

        # 致命缺失 → 强制观望
        if (not above_liq or above_liq <= 0) and (not below_liq or below_liq <= 0) and direction != "neutral":
            _force_neutral(s, "清算数据缺失")
            return True, ""
        if atr_15m <= 0 or mark_price <= 0:
            if direction != "neutral":
                _force_neutral(s, "ATR 或价格缺失")
                return True, ""

        # 强锚点冲突 → 强制观望
        if abs(direction_bias) > 0.4 and direction != "neutral":
            if (direction_bias > 0 and direction == "short") or (direction_bias < 0 and direction == "long"):
                _force_neutral(s, f"方向与强锚点({direction_bias:.3f})冲突")
                return True, ""

    # neutral 信号清空价格字段
    if direction == "neutral":
        for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
            s[f] = 0
        s["position_size"] = "无"
        if not s.get("execution_plan"):
            s["execution_plan"] = "等待新的交易信号"
        return True, ""

    # 校验价格完整性
    for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
        val = s.get(f)
        if val is None or float(val) <= 0:
            return False, f"缺少或无效的 {f}"

    return True, ""


# ------------------- 构建策略提示词 -------------------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    if cross_symbol is None:
        if symbol == "BTC": cross_symbol = "ETH"
        elif symbol == "ETH": cross_symbol = "BTC"
        else: cross_symbol = "ETH"

    def safe_val(key, default=0.0, scale=1.0, fmt=".2f", null_val=None):
        raw = data.get(key, null_val)
        if raw is None or raw == null_val:
            return ("缺失", True)
        try:
            val = float(raw) * scale
            return (f"{val:{fmt}}", False)
        except (ValueError, TypeError):
            return ("缺失", True)

    # 跨币种上下文
    cross_context = ""
    if eth_data:
        cross_mark = eth_data.get('mark_price', 0)
        cross_context = f"""【{cross_symbol} 跨币种数据】
现价：{cross_mark:.2f}
清算比值：{eth_data.get('liq_ratio', 0):.2f}
OI 24h变化：{eth_data.get('oi_change_24h', 0):+.1f}%
CVD斜率：{eth_data.get('cvd_slope', 0):.4f}
顶多空分位：{eth_data.get('top_ls_percentile', 50):.0f}%"""

    # 提取所有字段（完整列表保持不变，此处省略重复的提取代码以保持清晰，实际代码会包含所有字段）
    mark_price_str, _ = safe_val('mark_price', fmt=".2f")
    atr_str, _ = safe_val('atr', fmt=".2f")
    price_percentile_str, _ = safe_val('price_percentile', 50.0, fmt=".0f")
    fear_greed = data.get('fear_greed', 50)
    lth_rp_str, _ = safe_val('lth_realized_price', fmt=".2f")
    sth_rp_str, _ = safe_val('sth_realized_price', fmt=".2f")
    sth_sopr_str, _ = safe_val('sth_sopr', 1.0, fmt=".3f")
    stable_trend_str, _ = safe_val('stablecoin_trend_7d', fmt="+.1f")
    oi_chg_str, _ = safe_val('oi_change_24h', fmt="+.1f")
    fund_perc_str, _ = safe_val('funding_percentile', 50.0, fmt=".0f")
    cvd_slope_str, _ = safe_val('cvd_slope', fmt=".4f")
    taker_str, _ = safe_val('taker_ratio_1h', fmt=".3f")
    nf24h_str, _ = safe_val('netflow_24h', scale=1/1e6, fmt=".1f")
    above_trigger = data.get('above_trigger', 'N/A')
    below_trigger = data.get('below_trigger', 'N/A')
    max_pain_str, _ = safe_val('max_pain', fmt=".2f")
    pressure_str, _ = safe_val('large_order_pressure', fmt=".3f")
    direction_bias = data.get('direction_bias', 0.0)
    basis_perc_str, _ = safe_val('basis_percentile', 50.0, fmt=".0f")
    pc_str, _ = safe_val('put_call_ratio', fmt=".4f")
    btc_dom_trend_str, _ = safe_val('btc_dominance_trend_7d', fmt="+.1f")
    borrow_str, _ = safe_val('borrow_rate', scale=100, fmt=".2f")

    # 缺失约束
    data_quality = data.get("data_quality", {})
    core_missing = [k for k in ["kline", "heatmap", "cvd"] if data_quality.get(k) == "❌ 缺失"]
    constraint_note = ""
    if core_missing:
        constraint_note = f"\n【重要约束】以下核心数据缺失：{', '.join(core_missing)}。你必须将置信度设为'低'，若清算数据缺失则必须输出'neutral'。\n"

    prompt = f"""你是一位拥有20年经验的加密货币交易员。你的任务是基于提供的所有市场数据，进行深度分析并输出最终交易策略。

【铁律】
1. 周期一致：价值区(价格≤LTH成本)只做多，派发区(价格≥STH成本*1.3且LTH SOPR>1.2)只做空。方向与周期矛盾时必须观望。
2. 投票纪律：三组方向投票（资金流、情绪、链上）必须至少两组一致，否则观望。
3. 盈亏比约束：非观望策略必须满足 (止盈-入场)/(入场-止损) ≥ 2:1，否则降为观望。
4. 锚点服从：若 |direction_bias| > 0.4 且方向相反，强制观望；否则可降仓执行。
5. 数据完整性：必须引用下方所有原始数据，不得遗漏任何字段。

【{symbol} 市场数据】
现价：{mark_price_str} (7日分位{price_percentile_str}%)
ATR(4h)：{atr_str}
恐慌贪婪：{fear_greed}
LTH成本：{lth_rp_str}，STH成本：{sth_rp_str}，STH SOPR：{sth_sopr_str}
稳定币市值趋势：{stable_trend_str}%
OI 24h变化：{oi_chg_str}%，资金费率分位：{fund_perc_str}%
CVD斜率：{cvd_slope_str}，主动买卖比(1h)：{taker_str}
24h期货净流：{nf24h_str}M
上方清算触发距：{above_trigger}点，下方：{below_trigger}点
期权最大痛点：{max_pain_str}，P/C比：{pc_str}
大单压迫比：{pressure_str}
合约基差分位：{basis_perc_str}%
BTC市值占比趋势：{btc_dom_trend_str}%
借贷利率：{borrow_str}%
{constraint_note}
系统锚点 direction_bias：{direction_bias:.3f}

{cross_context}

【分析框架】
你必须按以下结构完成分析，并最终输出JSON策略。

一、周期定位
- 基于LTH/STH成本、STH SOPR、恐慌贪婪、稳定币趋势，判断当前周期阶段（价值区/拉升区/派发区/下跌区）。
- 给出最大允许仓位上限。

二、三组方向投票（必须逐项填写数据及判断）
第一组：资金流方向
  - CVD斜率：{cvd_slope_str} → [多/空/中性]
  - 主动买卖比：{taker_str} → [多/空/中性]
  - 24h期货净流：{nf24h_str}M → [多/空/中性]
  资金流投票：[看多/看空/中性]

第二组：情绪方向
  - OI 24h变化：{oi_chg_str}% → [多/空/中性]
  - 资金费率分位：{fund_perc_str}% → [多/空/中性]
  - 恐慌贪婪：{fear_greed} → [多/空/中性]
  情绪投票：[看多/看空/中性]

第三组：链上方向
  - STH SOPR：{sth_sopr_str} → [多/空/中性]
  - 现价 vs STH成本：{mark_price_str} vs {sth_rp_str} → [强化/弱化]
  链上投票：[看多/看空/中性]

最终方向：[做多/做空/观望] (必须三组中至少两组一致，否则观望)

三、战术定位
- 入场区间：基于清算触发距、期权痛点、大单压迫比确定。
- 止损：基于ATR和关键支撑/阻力设置。
- 止盈：基于清算密集区或期权痛点。
- 盈亏比计算：(止盈-入场)/(入场-止损) = __:1 → 必须≥2:1或明确不满足。

四、逻辑终审
- 周期允许方向 vs 投票方向 → [一致/矛盾] → [处理]
- 盈亏比 → [满足/不满足]
- 锚点一致性 → [一致/冲突] → [处理]

【输出格式】(严格JSON，不要代码块标记)
{{
  "direction": "做多 / 做空 / 观望",
  "confidence": "高 / 中 / 低",
  "position_size": "重仓 / 中仓 / 轻仓 / 无",
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "execution_plan": "一句话指令",
  "reasoning": "包含完整的周期定位、三组投票详表、战术定位和逻辑终审过程",
  "risk_note": "核心风险（20字内）",
  "cycle_phase": "价值区 / 拉升区 / 派发区 / 下跌区",
  "risk_reward_ratio": 0.0
}}
"""
    return prompt


# ------------------- 首席交易员调用 -------------------
def call_trader(prompt: str, max_retries: int = MAX_RETRIES) -> dict:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=TIMEOUT_SECONDS
    )
    for attempt in range(max_retries):
        try:
            logger.info(f"首席交易员 调用 (尝试 {attempt+1}/{max_retries})")
            resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                timeout=TIMEOUT_SECONDS,
                stop=["}\n```"]
            )
            content = resp.choices[0].message.content or ""
            reasoning = getattr(resp.choices[0].message, 'reasoning_content', None)
            _log_response(prompt, content, reasoning)

            final_content = content.strip() if content else (reasoning or "")
            if not final_content:
                raise ValueError("响应内容为空")

            json_str = extract_json(final_content)
            s = json.loads(json_str)

            # 方向标准化
            dir_map = {"做多": "long", "做空": "short", "观望": "neutral",
                       "long": "long", "short": "short", "neutral": "neutral"}
            s["direction"] = dir_map.get(s.get("direction", ""), "neutral")

            # 仓位标准化
            pos_map = {"轻仓": "light", "中仓": "medium", "重仓": "heavy", "无": "none",
                       "light": "light", "medium": "medium", "heavy": "heavy", "none": "none"}
            s["position_size"] = pos_map.get(s.get("position_size", ""), "none")

            # 置信度标准化
            conf_map = {"高": "high", "中": "medium", "低": "low",
                        "high": "high", "medium": "medium", "low": "low"}
            s["confidence"] = conf_map.get(s.get("confidence", ""), "medium")

            s.setdefault("reasoning", "")
            s.setdefault("risk_note", "")
            s.setdefault("execution_plan", "")
            s["_model_used"] = resp.model
            return s

        except Exception as e:
            logger.warning(f"首席交易员调用失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                return {
                    "direction": "neutral", "confidence": "低", "position_size": "none",
                    "entry_price_low": 0, "entry_price_high": 0, "stop_loss": 0, "take_profit": 0,
                    "execution_plan": "模型调用失败", "reasoning": "调用失败", "risk_note": "",
                    "cycle_phase": "未知", "risk_reward_ratio": 0.0, "_model_used": "fallback"
                }