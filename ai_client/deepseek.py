"""
deepseek.py — 三角色闭环（顶级交易员 · 审计官 · 交易委员会）
顶级交易员：清算动力学 + 多空博弈 + 预期差分析（每步数据填空 → 定性分析）
审计官：逐条审计，严重性评级（[严重/中等/轻度]）
交易委员会：独立裁决，可驳回指控，最终决定策略
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
def _log_response(prompt: str, content: str, reasoning: str = None):
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/deepseek_{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "content": content, "reasoning": reasoning}, f, ensure_ascii=False, indent=2)
    except:
        pass


def extract_json(content: str) -> str:
    """增强的 JSON 提取器，支持代码块包裹和纯 JSON"""
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
    direction = s.get("direction")
    if direction not in ["long", "short", "neutral"]:
        return False, f"无效方向: {direction}"

    if data:
        atr_15m = data.get("atr_15m", 0)
        mark_price = data.get("mark_price", 0)
        above_liq = data.get("above_liq", 0)
        below_liq = data.get("below_liq", 0)
        direction_bias = data.get("direction_bias", 0.0)

        if (not above_liq or above_liq <= 0) and (not below_liq or below_liq <= 0) and direction != "neutral":
            _force_neutral(s, "清算数据缺失")
            return True, ""
        if atr_15m <= 0 or mark_price <= 0:
            if direction != "neutral":
                _force_neutral(s, "ATR 或价格缺失")
                return True, ""

        if abs(direction_bias) > 0.4 and direction != "neutral":
            if (direction_bias > 0 and direction == "short") or (direction_bias < 0 and direction == "long"):
                _force_neutral(s, f"方向与强锚点({direction_bias:.3f})冲突")
                return True, ""

    if direction == "neutral":
        for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
            s[f] = 0
        s["position_size"] = "无"
        if not s.get("execution_plan"):
            s["execution_plan"] = "等待触发条件"
        return True, ""

    for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
        val = s.get(f)
        if val is None or float(val) <= 0:
            return False, f"缺少或无效的 {f}"

    return True, ""


# ------------------- 交易员提示词 -------------------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    if cross_symbol is None:
        cross_symbol = "ETH" if symbol == "BTC" else "BTC"

    def safe_val(key, default=0.0, scale=1.0, fmt=".2f"):
        raw = data.get(key)
        if raw is None:
            return ("缺失", True)
        try:
            val = float(raw) * scale
            return (f"{val:{fmt}}", False)
        except (ValueError, TypeError):
            return ("缺失", True)

    # ========== 提取所有字段（完整列表保持不变） ==========
    mark_price_str, _ = safe_val('mark_price', fmt=".2f")
    atr_str, _ = safe_val('atr', fmt=".2f")
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
    above_liq_str, _ = safe_val('above_liq', scale=1/1e9, fmt=".2f")
    below_liq_str, _ = safe_val('below_liq', scale=1/1e9, fmt=".2f")
    liq_ratio_str, _ = safe_val('liq_ratio', fmt=".2f")
    above_trigger = data.get('above_trigger', 'N/A')
    below_trigger = data.get('below_trigger', 'N/A')
    large_sell_str, _ = safe_val('large_sell_value', scale=1/1e6, fmt=".1f")
    large_buy_str, _ = safe_val('large_buy_value', scale=1/1e6, fmt=".1f")
    pressure_str, _ = safe_val('large_order_pressure', fmt=".3f")
    ob_imbalance_str, _ = safe_val('orderbook_imbalance', fmt=".3f")
    lure_str, _ = safe_val('lure_risk_factor', fmt=".2f")
    max_pain_str, _ = safe_val('max_pain', fmt=".2f")
    basis_perc_str, _ = safe_val('basis_percentile', 50.0, fmt=".0f")
    pc_str, _ = safe_val('put_call_ratio', fmt=".4f")
    btc_dom_trend_str, _ = safe_val('btc_dominance_trend_7d', fmt="+.1f")
    borrow_str, _ = safe_val('borrow_rate', scale=100, fmt=".2f")
    exchange_btc_str, _ = safe_val('exchange_btc_change_24h', fmt="+.0f")
    spot_24h_str, _ = safe_val('spot_netflow_24h', scale=1/1e6, fmt=".1f")
    spot_div_str, _ = safe_val('spot_vs_futures_divergence', fmt=".2f")
    top_ls_perc_str, _ = safe_val('top_ls_percentile', 50.0, fmt=".0f")
    eth_btc_perc_str, _ = safe_val('eth_btc_percentile', 50.0, fmt=".0f")
    price_percentile_str, _ = safe_val('price_percentile', 50.0, fmt=".0f")
    vol_factor_str, _ = safe_val('vol_factor', 1.0, fmt=".2f")
    cgdi_perc_str, _ = safe_val('cgdi_percentile', 50.0, fmt=".0f")
    direction_bias = data.get('direction_bias', 0.0)

    # 跨币种上下文
    cross_context = ""
    if eth_data:
        cross_liq = eth_data.get('liq_ratio', 0)
        cross_cvd = eth_data.get('cvd_slope', 0)
        cross_oi = eth_data.get('oi_change_24h', 0)
        cross_context = f"""
跨币种数据（{cross_symbol}）：清算比值 {cross_liq:.2f}，CVD斜率 {cross_cvd:.4f}，OI 24h变化 {cross_oi:+.1f}%"""

    # 缺失约束
    data_quality = data.get("data_quality", {})
    core_missing = [k for k in ["kline", "heatmap", "cvd"] if data_quality.get(k) == "❌ 缺失"]
    constraint_note = ""
    if core_missing:
        constraint_note = f"\n【重要约束】核心数据缺失：{', '.join(core_missing)}。置信度强制设为'低'，若清算数据缺失则输出'neutral'。"

    prompt = f"""你是一位拥有15年经验的加密货币顶级交易员，精通清算动力学、多空博弈和预期差分析。

【核心铁律】
1. 每步必须先完成「数据确认」填空，再进入定性分析。
2. 盈亏比必须≥2:1，否则降为观望。
3. 最终方向与系统锚点direction_bias={direction_bias:.3f}冲突且|bias|>0.4时，强制观望。
4. 数据确认表中必须填写所有字段，不可跳过。

【市场数据】
现价：{mark_price_str}，ATR(4h)：{atr_str}
恐慌贪婪：{fear_greed}
LTH成本：{lth_rp_str}，STH成本：{sth_rp_str}，STH SOPR：{sth_sopr_str}
稳定币市值趋势：{stable_trend_str}%
OI 24h变化：{oi_chg_str}%，资金费率分位：{fund_perc_str}%
CVD斜率：{cvd_slope_str}，主动买卖比(1h)：{taker_str}
24h期货净流：{nf24h_str}M，现货24h净流：{spot_24h_str}M
现货/期货背离度：{spot_div_str}
上方清算：{above_liq_str}B，触发距{above_trigger}点
下方清算：{below_liq_str}B，触发距{below_trigger}点
清算比值：{liq_ratio_str}
大额卖单：{large_sell_str}M美元，买单：{large_buy_str}M美元，压迫比：{pressure_str}
订单簿失衡率：{ob_imbalance_str}，诱饵风险系数：{lure_str}
期权最大痛点：{max_pain_str}，P/C比：{pc_str}
合约基差分位：{basis_perc_str}%
BTC市值占比趋势：{btc_dom_trend_str}%
ETH/BTC汇率分位：{eth_btc_perc_str}%
借贷利率：{borrow_str}%
交易所BTC余额变化：{exchange_btc_str} BTC
价格7日分位：{price_percentile_str}%，波动因子：{vol_factor_str}
CGDI分位：{cgdi_perc_str}%，大户多空比分位：{top_ls_perc_str}%
{cross_context}
{constraint_note}

请严格按照以下五步进行分析，每步必须先完成数据确认填空，再进入定性分析。

【第一步：清算动力学分析】
数据确认表：
| 字段 | 当前值 | 意味着什么？|
|------|--------|-----------|
| 上方清算触发距 | {above_trigger}点 | |
| 下方清算触发距 | {below_trigger}点 | |
| 清算比值 | {liq_ratio_str} | |
| 大单压迫比 | {pressure_str} | |
| 订单簿失衡率 | {ob_imbalance_str} | |
| 诱饵风险系数 | {lure_str} | |
| 期权最大痛点 | {max_pain_str} | |
| ATR(4h) | {atr_str} | |
完成以下问题：
1. 做市商最可能穿刺的方向是哪边？穿刺成本如何？
2. 该方向上的大单墙是否构成真实阻力？
3. 期权痛点的磁吸效应是强化还是削弱穿刺方向？
4. 是否存在"穿刺-反杀"的诱饵结构？
5. 清算维度结论：[向上穿刺/向下穿刺/均衡]

【第二步：多空博弈分析】
多头数据确认表（至少5条）：
| 字段 | 当前值 | 为什么支持做多？|
|------|--------|----------------|
| CVD斜率 | {cvd_slope_str} | |
| 主动买卖比(1h) | {taker_str} | |
| 24h期货净流 | {nf24h_str}M | |
| STH SOPR | {sth_sopr_str} | |
| 现价 vs STH成本 | {mark_price_str} vs {sth_rp_str} | |
| 恐慌贪婪指数 | {fear_greed} | |
| （可补充跨币种数据） | | |
多头论据（引用上表）：
1. 
2. 
3. 

空头数据确认表（至少5条）：
| 字段 | 当前值 | 为什么支持做空？|
|------|--------|----------------|
| OI 24h变化 | {oi_chg_str}% | |
| 资金费率分位 | {fund_perc_str}% | |
| 大户多空比分位 | {top_ls_perc_str}% | |
| 现货24h净流 | {spot_24h_str}M | |
| 交易所BTC余额变化 | {exchange_btc_str} BTC | |
| 现货/期货背离度 | {spot_div_str} | |
| （可补充跨币种数据） | | |
空头论据（引用上表）：
1. 
2. 
3. 

交叉质询：
- 多头攻击空头最脆弱的一环：
- 空头攻击多头最脆弱的一环：
博弈维度结论：[多头优势/空头优势/僵持]

【第三步：预期差分析】
数据确认表：
| 字段 | 当前值 | 意味着什么？|
|------|--------|-----------|
| 恐慌贪婪指数 | {fear_greed} | |
| 合约基差分位 | {basis_perc_str}% | |
| 稳定币市值趋势 | {stable_trend_str}% | |
| 交易所BTC余额变化 | {exchange_btc_str} BTC | |
| 借贷利率 | {borrow_str}% | |
| BTC市值占比趋势 | {btc_dom_trend_str}% | |
| P/C比 | {pc_str} | |
| 价格7日分位 | {price_percentile_str}% | |
| 波动因子 | {vol_factor_str} | |
| CGDI分位 | {cgdi_perc_str}% | |
回答：
1. 当前价格已定价了什么信息？
2. 存在什么未被定价的潜在变化？
3. 若价格打破均衡，谁会最意外？意外者的行动如何加速行情？
4. 预期差维度结论：[向上预期差/向下预期差/无显著预期差]

【第四步：三维汇聚与信号定性】
汇聚规则：
- 三方一致 → 重仓，置信度高
- 两方一致 → 中仓，置信度中
- 一方指向或无一致 → 观望

汇聚结果：
清算维度：[向上穿刺/向下穿刺/均衡] → 映射方向：[做多/做空/中性]
博弈维度：[多头优势/空头优势/僵持] → 映射方向：[做多/做空/中性]
预期差维度：[向上/向下/无] → 映射方向：[做多/做空/中性]
最终方向：[做多/做空/观望]
仓位：[重仓/中仓/轻仓/无]
置信度：[高/中/低]

信号定性（跨币种修正）：
- 跨币种清算方向与汇聚方向：[一致/矛盾/无数据]
- 定性：[系统性趋势/单币种独立行情/无法定性]
- 若为单币种独立行情，仓位自动降一级。
修正后仓位：[重仓/中仓/轻仓/无]

【第五步：价格路径推演与合约策略】
关键推演数据确认：
| 关键数据 | 当前值 | 对价格路径的影响 |
|----------|--------|-----------------|
| 最短穿刺方向 | （第一步） | |
| 清算密集区位置 | （第一步） | |
| 期权最大痛点 | {max_pain_str} | |
| 博弈结论 | （第二步） | |
| 预期差方向 | （第三步） | |
| ATR(4h) | {atr_str} | |

价格路径推演（必须整合流动性猎杀理论、行为金融学、博弈论）：
- 第一段运动：
- 第二段运动：
- 最终目标：

最终合约策略：
- 币种：{symbol}
- 方向：[做多/做空/观望]
- 现价：
- 仓位：[重仓/中仓/轻仓/无] (依据)
- 置信度：[高/中/低] (依据)
- 入场区间：[__-__] (依据)
- 止损：[__] (依据：设于败方逻辑成立位置之外)
- 止盈：[__] (依据：清算密集区/期权痛点)
- 盈亏比计算：(止盈-入场)/(入场-止损) = __:1 [必须≥2:1]
- 说明：[一句话指令或观望触发条件]

【输出JSON格式】
{{
  "direction": "做多/做空/观望",
  "confidence": "高/中/低",
  "position_size": "重仓/中仓/轻仓/无",
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "execution_plan": "",
  "reasoning": "完整的五步分析过程，包含所有数据确认表和定性分析",
  "risk_note": "",
  "cycle_phase": "",
  "risk_reward_ratio": 0.0,
  "vote_result": {{"清算维度": "", "博弈维度": "", "预期差维度": "", "一致组数": 0, "最终方向": ""}}
}}
"""
    return prompt


# ------------------- 首席交易员调用 -------------------
def call_trader(prompt: str) -> dict:
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                timeout=TIMEOUT_SECONDS,
                stop=["}\n```"]
            )
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip():
                raise ValueError("空响应")
            json_str = extract_json(content)
            s = json.loads(json_str)
            dir_map = {"做多": "long", "做空": "short", "观望": "neutral"}
            s["direction"] = dir_map.get(s.get("direction", ""), "neutral")
            pos_map = {"重仓": "heavy", "中仓": "medium", "轻仓": "light", "无": "none"}
            s["position_size"] = pos_map.get(s.get("position_size", ""), "none")
            conf_map = {"高": "high", "中": "medium", "低": "low"}
            s["confidence"] = conf_map.get(s.get("confidence", ""), "medium")
            s.setdefault("reasoning", "")
            s.setdefault("risk_note", "")
            s.setdefault("execution_plan", "")
            s["_model_used"] = resp.model
            return s
        except Exception as e:
            logger.warning(f"交易员调用失败: {e}")
            if attempt == MAX_RETRIES - 1:
                return {"direction": "neutral", "confidence": "低", "position_size": "none",
                        "entry_price_low": 0, "entry_price_high": 0, "stop_loss": 0, "take_profit": 0,
                        "execution_plan": "调用失败", "reasoning": "调用失败", "risk_note": "", "_model_used": "fallback"}
            time.sleep(RETRY_BASE_WAIT ** (attempt + 1))


# ------------------- 审计官 -------------------
def build_reviewer_prompt(strategy: dict, data: dict, symbol: str) -> str:
    direction_bias = data.get('direction_bias', 0.0)
    return f"""你是风控审计官。请逐条核查以下内容，并按模板输出审计报告。

【交易标的】{symbol}
【系统锚点】direction_bias={direction_bias:.3f}

【交易员策略摘要】
方向：{strategy.get('direction')}
仓位：{strategy.get('position_size')}
入场：{strategy.get('entry_price_low')}-{strategy.get('entry_price_high')}
止损：{strategy.get('stop_loss')}，止盈：{strategy.get('take_profit')}

【交易员完整推演】
{strategy.get('reasoning', '无')}

【审计规则】
按以下19条逐条核查，输出格式如下：
一、遗漏指标与分析缺失
- [发现或"已覆盖所有应分析的关键指标"]
二、数据与解读错误
- [发现或"未发现数据或解读错误"]
三、逻辑错误
- [发现或"未发现明显逻辑错误"]
四、关键反证提示
- [发现或"未发现关键反证被忽略"]
五、博弈层面审视
- [发现或"未发现博弈层面问题"]

每条发现的格式必须严格：
- 在[步骤X/决策点]中，交易员[具体问题]。该指标显示[具体数值/信号]，若纳入分析将[强化/削弱/推翻]当前方向判断。 [严重性：高/中/低]

【极其重要】你必须只返回纯JSON，不要加任何额外的文本、解释或代码块标记。JSON必须包含以下字段：
{{
  "verdict": "通过/存疑/驳回",
  "max_severity": "严重/中等/轻度/无",
  "severity_counts": {{"严重":0,"中等":0,"轻度":0}},
  "full_report": "完整审计文本，每条发现以换行分隔"
}}
"""


def call_reviewer(strategy: dict, data: dict, symbol: str) -> dict:
    prompt = build_reviewer_prompt(strategy, data, symbol)
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                timeout=120
            )
            content = resp.choices[0].message.content or ""
            logger.info(f"审计官原始响应: {content[:500]}")
            _log_response(prompt, content)

            # 增强的JSON提取
            json_str = extract_json(content)
            rev = json.loads(json_str)
            rev["full_report"] = rev.get("full_report", str(rev))
            rev.setdefault("verdict", "通过")
            rev.setdefault("max_severity", "无")
            rev.setdefault("severity_counts", {})
            return {**rev, "_model": resp.model}
        except Exception as e:
            logger.warning(f"审计官调用失败 (尝试 {attempt+1}): {e}, 原始响应前500字符: {content[:500] if 'content' in dir() else '无'}")
            if attempt == MAX_RETRIES - 1:
                return {"verdict": "通过", "max_severity": "无", "severity_counts": {}, "full_report": "审计官调用失败，跳过审计", "_model": "fallback"}
            time.sleep(RETRY_BASE_WAIT ** (attempt + 1))


# ------------------- 交易委员会 -------------------
def build_judge_prompt(strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> str:
    direction_bias = data.get('direction_bias', 0.0)
    return f"""你是交易委员会主席，拥有最终裁决权。请基于审计报告和交易员策略，独立做出裁决。

【标的信息】{symbol}，现价：{data.get('mark_price', 0):.2f}
【系统锚点】direction_bias={direction_bias:.3f}

【交易员完整策略】
方向：{strategy.get('direction')}
仓位：{strategy.get('position_size')}
置信度：{strategy.get('confidence')}
入场区间：{strategy.get('entry_price_low')}-{strategy.get('entry_price_high')}
止损：{strategy.get('stop_loss')}
止盈：{strategy.get('take_profit')}
执行计划：{strategy.get('execution_plan')}
风险提示：{strategy.get('risk_note')}
推演：{strategy.get('reasoning', '无')}

【审计报告】
{reviewer_report.get('full_report', '无审计报告')}
审计结论：{reviewer_report.get('verdict', '未知')}
最高严重性：{reviewer_report.get('max_severity', '未知')}

【裁决规则】
- 若审计严重性为"严重"：必须推翻原策略，改为观望。
- 若审计严重性为"中等"：可推翻、修改执行或维持原判，需说明理由。
- 若审计严重性为"轻度"或"无"：可修改执行或维持原判。
- 你有权驳回审计官的指控，但必须提供数据依据。

【极其重要】你必须只返回纯JSON，不要加任何额外的文本、解释或代码块标记。JSON格式如下：
{{
  "final_verdict": "维持原判/修改执行/推翻",
  "final_direction": "long/short/neutral",
  "final_confidence": "高/中/低",
  "final_position_size": "heavy/medium/light/none",
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "execution_plan": "",
  "risk_note": "",
  "audit_adopted": true,
  "audit_max_severity": "严重/中等/轻度/无",
  "final_reasoning": "裁决理由，针对审计指控逐条回应"
}}
如果维持原判，价格字段必须填写交易员给出的原始数值，不能写0。
"""


def call_judge(strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    prompt = build_judge_prompt(strategy, reviewer_report, data, symbol)
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=REASONING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                timeout=120
            )
            content = resp.choices[0].message.content or ""
            logger.info(f"委员会原始响应: {content[:500]}")
            _log_response(prompt, content)
            json_str = extract_json(content)
            result = json.loads(json_str)

            # 字段标准化
            dir_map = {"做多": "long", "做空": "short", "观望": "neutral",
                       "long": "long", "short": "short", "neutral": "neutral"}
            result["final_direction"] = dir_map.get(result.get("final_direction", ""), "neutral")
            pos_map = {"重仓": "heavy", "中仓": "medium", "轻仓": "light", "无": "none",
                       "heavy": "heavy", "medium": "medium", "light": "light", "none": "none"}
            result["final_position_size"] = pos_map.get(result.get("final_position_size", ""), "none")
            conf_map = {"高": "high", "中": "medium", "低": "low",
                        "high": "high", "medium": "medium", "low": "low"}
            result["final_confidence"] = conf_map.get(result.get("final_confidence", ""), "medium")

            # 维持原判时继承原策略的价格字段
            if result.get("final_verdict") == "维持原判":
                result["entry_price_low"] = result.get("entry_price_low") or strategy.get("entry_price_low", 0)
                result["entry_price_high"] = result.get("entry_price_high") or strategy.get("entry_price_high", 0)
                result["stop_loss"] = result.get("stop_loss") or strategy.get("stop_loss", 0)
                result["take_profit"] = result.get("take_profit") or strategy.get("take_profit", 0)
                result["execution_plan"] = result.get("execution_plan") or strategy.get("execution_plan", "")
                result["risk_note"] = result.get("risk_note") or strategy.get("risk_note", "")
                result["final_confidence"] = result.get("final_confidence") or strategy.get("confidence", "中")
                result["final_position_size"] = result.get("final_position_size") or strategy.get("position_size", "none")

            return {**result, "_model": resp.model}
        except Exception as e:
            logger.warning(f"交易委员会调用失败 (尝试 {attempt+1}): {e}, 原始响应: {content[:500] if 'content' in dir() else '无'}")
            if attempt == MAX_RETRIES - 1:
                return {"final_verdict": "维持原判", "final_direction": strategy.get("direction", "neutral"),
                        "final_confidence": strategy.get("confidence", "低"),
                        "final_position_size": strategy.get("position_size", "none"),
                        "entry_price_low": strategy.get("entry_price_low", 0),
                        "entry_price_high": strategy.get("entry_price_high", 0),
                        "stop_loss": strategy.get("stop_loss", 0),
                        "take_profit": strategy.get("take_profit", 0),
                        "execution_plan": strategy.get("execution_plan", ""),
                        "risk_note": strategy.get("risk_note", ""),
                        "audit_adopted": False, "audit_max_severity": "无",
                        "final_reasoning": "委员会调用失败，自动维持原判", "_model": "fallback"}
            time.sleep(RETRY_BASE_WAIT ** (attempt + 1))


def apply_final_verdict(strategy: dict, judge_result: dict) -> dict:
    """应用委员会裁决，将最终策略写入strategy"""
    verdict = judge_result.get("final_verdict", "维持原判")
    logger.info(f"应用最终决议: {verdict}")

    # 正常化委员会返回的字段
    strategy["_judge_verdict"] = verdict
    strategy["_judge_reasoning"] = judge_result.get("final_reasoning", "")

    if verdict == "推翻":
        direction = judge_result.get("final_direction", "neutral")
        if direction == "neutral":
            _force_neutral(strategy, "委员会推翻并改为观望")
        else:
            strategy["direction"] = direction
            strategy["confidence"] = judge_result.get("final_confidence", "中")
            strategy["position_size"] = judge_result.get("final_position_size", "none")
            strategy["entry_price_low"] = judge_result.get("entry_price_low", 0) or 0
            strategy["entry_price_high"] = judge_result.get("entry_price_high", 0) or 0
            strategy["stop_loss"] = judge_result.get("stop_loss", 0) or 0
            strategy["take_profit"] = judge_result.get("take_profit", 0) or 0
            strategy["execution_plan"] = judge_result.get("execution_plan", "")
            strategy["risk_note"] = judge_result.get("risk_note", "")
        return strategy

    elif verdict == "修改执行":
        # 只覆盖委员会有值的字段
        for k in ["final_confidence", "final_position_size", "entry_price_low", "entry_price_high", "stop_loss", "take_profit", "execution_plan"]:
            if k in judge_result and judge_result[k]:
                strategy[k] = judge_result[k]
        if "risk_note" in judge_result and judge_result["risk_note"]:
            strategy["risk_note"] = judge_result["risk_note"]
        if "final_direction" in judge_result and judge_result["final_direction"] != strategy.get("direction"):
            strategy["direction"] = judge_result["final_direction"]
        return strategy

    else:  # 维持原判
        # 委员会可能返回了新字段，但维持原判时原则上保留原策略
        strategy["risk_note"] = judge_result.get("risk_note") or strategy.get("risk_note", "")
        strategy["execution_plan"] = judge_result.get("execution_plan") or strategy.get("execution_plan", "")
        return strategy
