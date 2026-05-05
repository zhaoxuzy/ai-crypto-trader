"""
deepseek.py — 生产级三角色闭环（最终修正版 - 修复三个细微缺陷）
- 标准化映射函数清洗空格/小写
- safe_val 防御 inf/NaN
- 委员会 audit_max_severity 强制英文化
"""

import os, json, time, re
from datetime import datetime
from openai import OpenAI
from utils.logger import logger

TICK_SIZE = 0.1
MAX_RETRIES = 3
RETRY_BASE_WAIT = 2
TIMEOUT_SECONDS = 180

FAST_MODEL = "deepseek-v4-pro"
REASONING_MODEL = "deepseek-v4-pro"

# 合法值集合
VALID_DIRECTIONS = {"long", "short", "neutral"}
VALID_CONFIDENCES = {"high", "medium", "low"}
VALID_POSITION_SIZES = {"heavy", "medium", "light", "none"}

# ---------- 标准化映射函数 (缺陷1修复：清洗空白和大小写) ----------
def norm_direction(raw: str) -> str:
    if not raw: return "neutral"
    clean = raw.strip().lower()
    if clean in VALID_DIRECTIONS:
        return clean
    mapping = {"做多": "long", "做空": "short", "观望": "neutral"}
    return mapping.get(clean, "neutral")

def norm_confidence(raw: str) -> str:
    if not raw: return "medium"
    clean = raw.strip().lower()
    if clean in VALID_CONFIDENCES:
        return clean
    mapping = {"高": "high", "中": "medium", "低": "low"}
    return mapping.get(clean, "medium")

def norm_position_size(raw: str) -> str:
    if not raw: return "none"
    clean = raw.strip().lower()
    if clean in VALID_POSITION_SIZES:
        return clean
    mapping = {"重仓": "heavy", "中仓": "medium", "轻仓": "light", "无": "none", "none": "none"}
    return mapping.get(clean, "none")


# ---------- 辅助函数 ----------
def _log_response(role: str, prompt: str, content: str, reasoning: str = None):
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/{role}_{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "content": content, "reasoning": reasoning}, f, ensure_ascii=False, indent=2)
    except:
        pass


def extract_json_safe(content: str) -> str:
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
    raise ValueError("JSON 未闭合，不走暴力修补")


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
    if direction not in VALID_DIRECTIONS:
        return False, f"无效方向: {direction}"

    if data:
        atr_15m = data.get("atr_15m", 0)
        mark_price = data.get("mark_price", 0)
        above_liq = data.get("above_liq", 0)
        below_liq = data.get("below_liq", 0)
        bias_quality = data.get("_bias_quality", "reliable")
        direction_bias = data.get("direction_bias", 0.0)

        if (not above_liq or above_liq <= 0) and (not below_liq or below_liq <= 0) and direction != "neutral":
            _force_neutral(s, "清算数据缺失")
            return True, ""
        if atr_15m <= 0 or mark_price <= 0:
            if direction != "neutral":
                _force_neutral(s, "ATR 或价格缺失")
                return True, ""

        if bias_quality in ("reliable", "degraded") and abs(direction_bias) > 0.4 and direction != "neutral":
            if (direction_bias > 0 and direction == "short") or (direction_bias < 0 and direction == "long"):
                _force_neutral(s, f"方向与锚点({direction_bias:.3f})冲突")
                return True, ""

    if direction == "neutral":
        for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
            s[f] = 0
        s["position_size"] = "none"
        if not s.get("execution_plan"):
            s["execution_plan"] = "等待触发条件"
        return True, ""

    for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
        val = s.get(f)
        if val is None or float(val) <= 0:
            return False, f"缺少或无效的 {f}"

    return True, ""


# ---------- 统一微观数据时效注入 ----------
def _inject_ages(data: dict):
    now = time.time()
    keys_ts = {
        "ob_imbalance_ts": "ob_age",
        "taker_ratio_ts": "taker_age",
        "cvd_slope_ts": "cvd_age",
        "large_order_ts": "large_order_age",
        "liquidation_ts": "liq_age"
    }
    for ts_key, age_key in keys_ts.items():
        ts = data.get(ts_key)
        if ts and ts > 0:
            data[age_key] = now - ts
        else:
            data[age_key] = float('inf')


# ---------- 清算穿刺量化 ----------
def compute_liquidation_bias(data: dict) -> dict:
    liq_ratio = data.get('liq_ratio', 1.0)
    cvd_slope = data.get('cvd_slope', 0.0)
    taker_ratio = data.get('taker_ratio_1h', 0.5)
    ob_imbalance = data.get('orderbook_imbalance', 0.0)
    ob_age = data.get('ob_age', float('inf'))
    large_pressure = data.get('large_order_pressure', 0.0)
    max_pain = data.get('max_pain', 0.0)
    atr = data.get('atr', 0.0)
    mark = data.get('mark_price', 0.0)

    if ob_age > 30:
        ob_imbalance = 0.0

    score = (
        (liq_ratio - 1.0) * 0.4 +
        (1 if cvd_slope > 0 else -1) * 0.3 +
        (taker_ratio - 0.5) * 0.3
    )

    direction = 'balanced'
    if score > 0.15:
        direction = 'up'
    elif score < -0.15:
        direction = 'down'

    lure_flag = (direction == 'up' and large_pressure < -0.5) or (direction == 'down' and large_pressure > 0.5)

    pain_effect = False
    if atr > 0 and max_pain > 0 and abs(max_pain - mark) < 1.0 * atr:
        if (direction == 'up' and max_pain > mark) or (direction == 'down' and max_pain < mark):
            pain_effect = True

    return {
        'puncture_direction': direction,
        'puncture_score': score,
        'lure_risk': lure_flag,
        'pain_magnet': pain_effect
    }


# ---------- 微观数据时效评估 ----------
def assess_micro_quality(data: dict) -> dict:
    checks = {
        "orderbook_fresh": data.get("ob_age", float('inf')) < 30,
        "taker_fresh": data.get("taker_age", float('inf')) < 60,
        "cvd_fresh": data.get("cvd_age", float('inf')) < 300,
        "large_order_fresh": data.get("large_order_age", float('inf')) < 300,
        "liquidation_fresh": data.get("liq_age", float('inf')) < 600,
    }
    fresh_count = sum(checks.values())
    overall = "good" if fresh_count >= 4 else ("degraded" if fresh_count >= 2 else "poor")
    return {**checks, "overall": overall}


# ---------- 预期定价仪表盘 ----------
def build_expectation_dashboard(data: dict) -> str:
    basis_annual = data.get('basis_annualized', 0)
    basis_median = data.get('basis_median', 8)
    funding_pct = data.get('funding_percentile', 50)
    cgdi_pct = data.get('cgdi_percentile', 50)
    stable_flow = data.get('stablecoin_trend_7d', 0)
    btc_dom = data.get('btc_dominance_trend_7d', 0)
    borrow = data.get('borrow_rate', 0) * 100
    pc = data.get('put_call_ratio', 1.0)
    price_pct = data.get('price_percentile', 50)
    vol_factor = data.get('vol_factor', 1.0)

    return f"""【预期定价仪表盘】
| 指标 | 当前值 | 历史基线 | 市场定价了什么？ |
|------|--------|----------|----------------|
| 3月基差年化 | {basis_annual:.1f}% | {basis_median:.1f}% | 若>>基线：期货溢价过热 |
| 资金费率分位 | {funding_pct:.0f}% | 50% | 多头支付意愿强度 |
| CGDI分位 | {cgdi_pct:.0f}% | 50% | 综合市场贪婪度 |
| 稳定币净流7d | {stable_flow:+.1f}% | 基线+0.5% | 资金面松紧 |
| BTC.D趋势7d | {btc_dom:+.1f}% | 0% | 风险偏好 |
| 借贷利率 | {borrow:.2f}% | 历史均值 | 杠杆资金紧张度 |
| P/C比 | {pc:.3f} | 0.7 | 对冲需求，>1 示恐慌 |
| 价格7日分位 | {price_pct:.0f}% | 50% | 超买/超卖程度 |
| 波动因子 | {vol_factor:.2f} | 1.0 | 不确定性定价 |

预期差分析必须回答：
1. 市场定价最极端的方向（贪婪/恐惧）是什么？依据仪表盘哪些指标？
2. 找出与极端定价矛盾的两个指标，这构成潜在“预期差”。
3. 若价格突然朝矛盾方向移动 1 ATR，谁会最意外？
4. 结论：预期差方向（向上/向下/无）必须基于上述矛盾证据，不可猜测。"""


# ------------------- 构建策略提示词 -------------------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    if cross_symbol is None:
        cross_symbol = "ETH" if symbol == "BTC" else "BTC"

    _inject_ages(data)

    # 缺陷2修复：safe_val 防御inf/NaN
    def safe_val(key, default=0.0, scale=1.0, fmt=".2f"):
        raw = data.get(key)
        if raw is None:
            return ("[N/A]", True)
        try:
            val = float(raw) * scale
        except (ValueError, TypeError):
            return ("[N/A]", True)
        if val != val or abs(val) == float('inf'):  # NaN or inf
            return ("[N/A]", True)
        try:
            return (f"{val:{fmt}}", False)
        except ValueError:
            return ("[N/A]", True)

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
    pc_str, _ = safe_val('put_call_ratio', fmt=".4f")
    basis_perc_str, _ = safe_val('basis_percentile', 50.0, fmt=".0f")
    btc_dom_trend_str, _ = safe_val('btc_dominance_trend_7d', fmt="+.1f")
    borrow_str, _ = safe_val('borrow_rate', scale=100, fmt=".2f")
    exchange_btc_str, _ = safe_val('exchange_btc_change_24h', fmt="+.0f")
    spot_24h_str, _ = safe_val('spot_netflow_24h', scale=1/1e6, fmt=".1f")
    spot_div_str, _ = safe_val('spot_vs_futures_divergence', fmt=".2f")
    top_ls_perc_str, _ = safe_val('top_ls_percentile', 50.0, fmt=".0f")
    price_percentile_str, _ = safe_val('price_percentile', 50.0, fmt=".0f")
    vol_factor_str, _ = safe_val('vol_factor', 1.0, fmt=".2f")
    cgdi_perc_str, _ = safe_val('cgdi_percentile', 50.0, fmt=".0f")
    direction_bias = data.get('direction_bias', 0.0)

    puncture = compute_liquidation_bias(data)
    micro_q = assess_micro_quality(data)
    dashboard = build_expectation_dashboard(data)

    cross_context = ""
    if eth_data:
        cross_context = f"跨币种（{cross_symbol}）：清算比值 {eth_data.get('liq_ratio',0):.2f}，CVD斜率 {eth_data.get('cvd_slope',0):.4f}，OI变化 {eth_data.get('oi_change_24h',0):+.1f}%"

    core_missing = [k for k in ["kline","heatmap","cvd"] if data.get("data_quality",{}).get(k) == "❌ 缺失"]
    constraint_note = f"【重要约束】核心数据缺失：{', '.join(core_missing)}。置信度强制设为'低'，若清算数据缺失则输出'neutral'。" if core_missing else ""

    prompt = f"""你是一位拥有 15 年实战经验的加密货币首席交易员，专精于清算动力学、多空博弈定位与预期差分析。
【你的任务】
- 严格遵循五步分析框架，给出逻辑自洽的策略推演与具体交易计划。
- 遇到 [N/A] 标记的数据时，该维度不得作为判断依据，且整体置信度必须为低。
- reasoning 总字数 ≤ 3000 字，结构清晰、分层明确。
- 最终输出必须为纯 JSON，且所有枚举值使用中文（做多/做空/观望等）。

【系统预判】
清算穿刺预判：方向 {puncture['puncture_direction']}，得分 {puncture['puncture_score']:.2f}。诱饵风险：{puncture['lure_risk']}，期权磁吸：{puncture['pain_magnet']}。请在此基础上定性验证，允许推翻但必须给出数据反驳。
微观数据新鲜度：{micro_q['overall']}。若为 poor，高频信号（CVD、taker、OB）权重降为0，只能依赖中频数据。

{dashboard}

【市场数据】
现价：{mark_price_str}，ATR：{atr_str}，恐慌贪婪：{fear_greed}
LTH成本：{lth_rp_str}，STH成本：{sth_rp_str}，STH SOPR：{sth_sopr_str}
稳定币趋势：{stable_trend_str}%，OI 24h变化：{oi_chg_str}%，费率分位：{fund_perc_str}%
CVD斜率：{cvd_slope_str}，主动买卖比(1h)：{taker_str}
24h期货净流：{nf24h_str}M，现货24h净流：{spot_24h_str}M，背离度：{spot_div_str}
上方清算：{above_liq_str}B，触发距{above_trigger}点
下方清算：{below_liq_str}B，触发距{below_trigger}点，比值：{liq_ratio_str}
大单卖：{large_sell_str}M，买：{large_buy_str}M，压迫比：{pressure_str}
订单簿失衡率：{ob_imbalance_str}，诱饵风险：{lure_str}
期权痛点：{max_pain_str}，P/C比：{pc_str}，基差分位：{basis_perc_str}%
BTC.D趋势：{btc_dom_trend_str}%，借贷利率：{borrow_str}%
交易所BTC余额变化：{exchange_btc_str} BTC
价格7日分位：{price_percentile_str}%，波动因子：{vol_factor_str}，CGDI分位：{cgdi_perc_str}%
{cross_context}
{constraint_note}

严格按五步分析（每步均需数据确认表 + 定性分析），方向锚点 direction_bias={direction_bias:.3f}，冲突且锚点可信时必须观望。盈亏比 ≥ 2:1。

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
  "reasoning": "五步分析，总字数不超过3000字。",
  "risk_note": "",
  "risk_reward_ratio": 0.0,
  "vote_result": {{"清算维度": "", "博弈维度": "", "预期差维度": "", "一致组数": 0, "最终方向": ""}}
}}
"""
    return prompt


# ------------------- 首席交易员 -------------------
def call_trader(prompt: str) -> dict:
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=FAST_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=16384, timeout=TIMEOUT_SECONDS)
            content = resp.choices[0].message.content or ""
            _log_response("trader", prompt, content)
            if not content.strip(): raise ValueError("空响应")
            json_str = extract_json_safe(content)
            s = json.loads(json_str)
            s["direction"] = norm_direction(s.get("direction", ""))
            s["position_size"] = norm_position_size(s.get("position_size", ""))
            s["confidence"] = norm_confidence(s.get("confidence", ""))
            s.setdefault("reasoning",""); s.setdefault("risk_note",""); s.setdefault("execution_plan","")
            s["_model_used"] = resp.model
            return s
        except Exception as e:
            logger.warning(f"交易员调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {"direction":"neutral","confidence":"low","position_size":"none","entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,"execution_plan":"调用失败","reasoning":"调用失败","risk_note":"","_model_used":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))


# ------------------- 审计官 -------------------
def call_reviewer(strategy: dict, data: dict, symbol: str) -> dict:
    direction_bias = data.get('direction_bias', 0.0)
    prompt = f"""你是一位独立的风险审计官，负责对首席交易员的策略进行无偏见的严格审计。
【你的职责】
- 对照市场数据，逐项核查交易员分析中的遗漏、数据误用、逻辑断裂和反证缺失。
- 所有发现必须按“步骤/问题/数据证据/影响/严重性”格式记录。
- 最终裁决（通过/存疑/驳回）必须仅基于发现的严重性和数量，不受交易员声望影响。
- 输出必须为纯 JSON，包含完整的审计报告和严重性统计。

【标的】{symbol}【锚点】direction_bias={direction_bias:.3f}
【策略】方向：{strategy.get('direction')}，仓位：{strategy.get('position_size')}
入场：{strategy.get('entry_price_low')}-{strategy.get('entry_price_high')}
止损：{strategy.get('stop_loss')}，止盈：{strategy.get('take_profit')}
【推演】{strategy.get('reasoning', '无')}

按五节模板输出审计报告。
每条发现格式：在[步骤X]中，交易员[问题]。该指标显示[N/A]，若纳入将[强化/削弱/推翻]判断。[严重性：高/中/低]
提供 max_severity 和 severity_counts。
只输出纯JSON。

【输出JSON】
{{
  "verdict": "通过/存疑/驳回",
  "max_severity": "严重/中等/轻度/无",
  "severity_counts": {{"严重":0,"中等":0,"轻度":0}},
  "full_report": "完整审计文本"
}}
"""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=FAST_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=4096, timeout=120)
            content = resp.choices[0].message.content or ""
            json_str = extract_json_safe(content)
            rev = json.loads(json_str)
            rev["full_report"] = rev.get("full_report", str(rev))
            rev.setdefault("verdict", "驳回"); rev.setdefault("max_severity", "严重"); rev.setdefault("severity_counts", {})
            return {**rev, "_model": resp.model}
        except Exception as e:
            logger.warning(f"审计官调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {"verdict":"驳回","max_severity":"严重","severity_counts":{"严重":1},"full_report":"审计官调用失败，自动驳回","_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))


# ------------------- 交易委员会 -------------------
def call_judge(strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    direction_bias = data.get('direction_bias', 0.0)
    prompt = f"""你是交易委员会主席，拥有最终决策权。
【你的任务】
- 审议首席交易员的策略及审计官的完整报告。
- 对审计官的每一条严重指控必须逐条回应，明确采纳或驳斥的理由。
- 在三维汇聚存在分歧时，用你的市场经验作出最后平衡，但不得无视硬数据约束（如方向锚点、盈亏比底线）。
- 最终输出必须为纯 JSON，裁决字段（final_direction、final_confidence等）优先使用英文值（long/short/neutral、high/medium/low）。

【标的】{symbol}，现价：{data.get('mark_price', 0):.2f}，锚点：{direction_bias:.3f}

【交易员策略】
方向：{strategy.get('direction')}，仓位：{strategy.get('position_size')}，置信度：{strategy.get('confidence')}
入场：{strategy.get('entry_price_low')}-{strategy.get('entry_price_high')}
止损：{strategy.get('stop_loss')}，止盈：{strategy.get('take_profit')}
推演：{strategy.get('reasoning', '无')}

【审计报告】
{reviewer_report.get('full_report', '无')}
审计结论：{reviewer_report.get('verdict', '未知')}，最高严重性：{reviewer_report.get('max_severity', '未知')}

严重指控成立必须推翻。final_reasoning 必须逐条回应指控。维持原判时价格字段不能填0。
只输出纯JSON。

【输出JSON】
{{
  "final_verdict": "维持原判/修改执行/推翻",
  "final_direction": "long/short/neutral",
  "final_confidence": "high/medium/low",
  "final_position_size": "heavy/medium/light/none",
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "execution_plan": "",
  "risk_note": "",
  "audit_adopted": true,
  "audit_max_severity": "严重/中等/轻度/无",
  "final_reasoning": "裁决书正文"
}}
"""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=REASONING_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=16384, timeout=120)
            content = resp.choices[0].message.content or ""
            json_str = extract_json_safe(content)
            result = json.loads(json_str)
            # 标准化字段
            result["final_direction"] = norm_direction(result.get("final_direction", ""))
            result["final_position_size"] = norm_position_size(result.get("final_position_size", ""))
            result["final_confidence"] = norm_confidence(result.get("final_confidence", ""))
            
            # 缺陷3修复：audit_max_severity 强制英文化
            sev_map = {"严重": "critical", "中等": "medium", "轻度": "low", "无": "none"}
            raw_sev = result.get("audit_max_severity", "无")
            result["audit_max_severity"] = sev_map.get(raw_sev, raw_sev)
            
            # 维持原判时继承原策略字段
            if result.get("final_verdict") == "维持原判":
                result["entry_price_low"] = result.get("entry_price_low") or strategy.get("entry_price_low", 0)
                result["entry_price_high"] = result.get("entry_price_high") or strategy.get("entry_price_high", 0)
                result["stop_loss"] = result.get("stop_loss") or strategy.get("stop_loss", 0)
                result["take_profit"] = result.get("take_profit") or strategy.get("take_profit", 0)
                result["execution_plan"] = result.get("execution_plan") or strategy.get("execution_plan", "")
                result["risk_note"] = result.get("risk_note") or strategy.get("risk_note", "")
            result["final_reasoning"] = result.get("final_reasoning") or "裁决完成。"
            return {**result, "_model": resp.model}
        except Exception as e:
            logger.warning(f"委员会调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {
                    "final_verdict": "推翻",
                    "final_direction": "neutral",
                    "final_confidence": "low",
                    "final_position_size": "none",
                    "entry_price_low": 0,
                    "entry_price_high": 0,
                    "stop_loss": 0,
                    "take_profit": 0,
                    "execution_plan": "委员会调用失败，强制观望",
                    "risk_note": "系统故障",
                    "audit_adopted": False,
                    "audit_max_severity": "critical",
                    "final_reasoning": "委员会调用失败，自动推翻并观望",
                    "_model": "fallback"
                }
            time.sleep(RETRY_BASE_WAIT**(attempt+1))


def apply_final_verdict(strategy: dict, judge_result: dict) -> dict:
    verdict = judge_result.get("final_verdict", "维持原判")
    logger.info(f"应用最终决议: {verdict}")

    strategy["_judge_verdict"] = verdict
    strategy["_judge_reasoning"] = judge_result.get("final_reasoning", "")

    fields = ["direction", "confidence", "position_size", "entry_price_low", "entry_price_high", "stop_loss", "take_profit", "execution_plan", "risk_note"]

    if verdict == "推翻":
        if judge_result.get("final_direction") == "neutral":
            _force_neutral(strategy, "委员会推翻并改为观望")
        else:
            for k in fields:
                if k in judge_result and judge_result[k] is not None:
                    strategy[k] = judge_result[k]
        return strategy

    elif verdict == "修改执行":
        for k in fields:
            if k in judge_result and judge_result[k] is not None:
                strategy[k] = judge_result[k]
        return strategy

    else:  # 维持原判
        strategy["risk_note"] = judge_result.get("risk_note") or strategy.get("risk_note", "")
        strategy["execution_plan"] = judge_result.get("execution_plan") or strategy.get("execution_plan", "")
        return strategy
