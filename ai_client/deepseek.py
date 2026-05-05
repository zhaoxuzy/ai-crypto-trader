"""
deepseek.py — 三角色闭环（修复版）
修复：sv 函数参数名统一为 fmt，避免 TypeError
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

def _log_response(prompt, content, reasoning=None):
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/deepseek_{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "content": content, "reasoning": reasoning}, f, ensure_ascii=False, indent=2)
    except: pass

def extract_json(content: str) -> str:
    m = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if m: return m.group(1).strip()
    m = re.search(r'```\s*([\s\S]*?)\s*```', content)
    if m: return m.group(1).strip()
    start = content.find('{')
    if start == -1: raise ValueError("未找到 JSON")
    count = 0
    last_valid_end = -1
    for i, c in enumerate(content[start:], start):
        if c == '{': count += 1
        elif c == '}':
            count -= 1
            if count == 0: return content[start:i+1].strip()
            if count < 0: break
        if count == 0: last_valid_end = i
    if last_valid_end != -1:
        logger.warning("JSON 未闭合，已修补")
        return content[start:last_valid_end+1] + '}'
    logger.warning("JSON 损坏，暴力修补")
    return content[start:] + '}}'

def format_reasoning(text: str) -> str:
    if not text: return text
    if '\n' in text or '【' in text:
        return text.strip()
    text = re.sub(r'(?<=[。！？；：])\s*', '\n', text)
    return text.strip()

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
    if direction not in ["long", "short", "neutral"]: return False, f"无效方向: {direction}"
    if data:
        atr_15m = data.get("atr_15m", 0); mark_price = data.get("mark_price", 0)
        above_liq = data.get("above_liq", 0); below_liq = data.get("below_liq", 0)
        direction_bias = data.get("direction_bias", 0.0)
        if (not above_liq or above_liq <= 0) and (not below_liq or below_liq <= 0) and direction != "neutral":
            _force_neutral(s, "清算数据缺失"); return True, ""
        if atr_15m <= 0 or mark_price <= 0:
            if direction != "neutral": _force_neutral(s, "ATR 或价格缺失"); return True, ""
        if abs(direction_bias) > 0.4 and direction != "neutral":
            if (direction_bias > 0 and direction == "short") or (direction_bias < 0 and direction == "long"):
                _force_neutral(s, f"方向与强锚点({direction_bias:.3f})冲突"); return True, ""
    rr = s.get("risk_reward_ratio", 0)
    if rr > 0 and rr < 2.0 and direction != "neutral":
        s["risk_note"] = s.get("risk_note", "") + f" [系统提示] 盈亏比{rr:.1f}:1，偏低。"
        if s.get("confidence") == "high": s["confidence"] = "medium"
    if direction == "neutral":
        for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]: s[f] = 0
        s["position_size"] = "无"
        if not s.get("execution_plan"): s["execution_plan"] = "等待触发条件"
        return True, ""
    for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
        val = s.get(f)
        if val is None or float(val) <= 0: return False, f"缺少或无效的 {f}"
    return True, ""

# ------------------- 交易员提示词 -------------------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    if cross_symbol is None: cross_symbol = "ETH" if symbol == "BTC" else "BTC"
    
    # 修正：参数名从 f 改为 fmt
    def sv(key, default=0.0, scale=1.0, fmt=".2f"):
        raw = data.get(key)
        if raw is None: return ("缺失", True)
        try: return (f"{float(raw)*scale:{fmt}}", False)
        except: return ("缺失", True)

    mark_price_str, _ = sv('mark_price', fmt=".2f")
    atr_str, _ = sv('atr', fmt=".2f")
    fear_greed = data.get('fear_greed', 50)
    lth_rp_str, _ = sv('lth_realized_price', fmt=".2f")
    sth_rp_str, _ = sv('sth_realized_price', fmt=".2f")
    sth_sopr_str, _ = sv('sth_sopr', 1.0, fmt=".3f")
    stable_trend_str, _ = sv('stablecoin_trend_7d', fmt="+.1f")
    oi_chg_str, _ = sv('oi_change_24h', fmt="+.1f")
    fund_perc_str, _ = sv('funding_percentile', 50.0, fmt=".0f")
    cvd_slope_str, _ = sv('cvd_slope', fmt=".4f")
    taker_str, _ = sv('taker_ratio_1h', fmt=".3f")
    nf24h_str, _ = sv('netflow_24h', scale=1/1e6, fmt=".1f")
    above_liq_str, _ = sv('above_liq', scale=1/1e9, fmt=".2f")
    below_liq_str, _ = sv('below_liq', scale=1/1e9, fmt=".2f")
    liq_ratio_str, _ = sv('liq_ratio', fmt=".2f")
    above_trigger = data.get('above_trigger', 'N/A')
    below_trigger = data.get('below_trigger', 'N/A')
    large_sell_str, _ = sv('large_sell_value', scale=1/1e6, fmt=".1f")
    large_buy_str, _ = sv('large_buy_value', scale=1/1e6, fmt=".1f")
    pressure_str, _ = sv('large_order_pressure', fmt=".3f")
    ob_imbalance_str, _ = sv('orderbook_imbalance', fmt=".3f")
    lure_str, _ = sv('lure_risk_factor', fmt=".2f")
    max_pain_str, _ = sv('max_pain', fmt=".2f")
    basis_perc_str, _ = sv('basis_percentile', 50.0, fmt=".0f")
    pc_str, _ = sv('put_call_ratio', fmt=".4f")
    btc_dom_trend_str, _ = sv('btc_dominance_trend_7d', fmt="+.1f")
    borrow_str, _ = sv('borrow_rate', scale=100, fmt=".2f")
    exchange_btc_str, _ = sv('exchange_btc_change_24h', fmt="+.0f")
    spot_24h_str, _ = sv('spot_netflow_24h', scale=1/1e6, fmt=".1f")
    spot_div_str, _ = sv('spot_vs_futures_divergence', fmt=".2f")
    top_ls_perc_str, _ = sv('top_ls_percentile', 50.0, fmt=".0f")
    price_percentile_str, _ = sv('price_percentile', 50.0, fmt=".0f")
    vol_factor_str, _ = sv('vol_factor', 1.0, fmt=".2f")
    cgdi_perc_str, _ = sv('cgdi_percentile', 50.0, fmt=".0f")
    direction_bias = data.get('direction_bias', 0.0)

    cross_context = ""
    if eth_data:
        cross_context = f"跨币种（{cross_symbol}）：清算比值 {eth_data.get('liq_ratio',0):.2f}，CVD斜率 {eth_data.get('cvd_slope',0):.4f}，OI变化 {eth_data.get('oi_change_24h',0):+.1f}%"

    core_missing = [k for k in ["kline","heatmap","cvd"] if data.get("data_quality",{}).get(k) == "❌ 缺失"]
    constraint_note = f"【重要约束】核心数据缺失：{', '.join(core_missing)}。置信度强制设为'低'，若清算数据缺失则输出'neutral'。" if core_missing else ""

    prompt = f"""你是一位拥有15年经验的加密货币顶级交易员，精通清算动力学、多空博弈和预期差分析。

【核心铁律】
1. 每步必须先完成「数据确认」填空，再进入定性分析。
2. 盈亏比低于2:1时需在risk_note中提示，不再强制驳回，但置信度不能为高。
3. 最终方向与系统锚点direction_bias={direction_bias:.3f}冲突且|bias|>0.4时，强制观望。
4. 数据确认表中必须填写所有字段，不可跳过。
5. **reasoning字段必须按步骤分段，每段用换行分隔，严禁写成连续的一整段。**

【市场数据】
现价：{mark_price_str}，ATR(4h)：{atr_str}，恐慌贪婪：{fear_greed}
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

请严格按照以下五步进行分析，每步均需填写数据确认表，然后做定性分析。reasoning中必须保留所有分析过程。

【第一步：清算动力学分析】
数据确认表：（用简洁的键值对形式列出上方清算触发距、下方清算触发距、清算比值、大单压迫比、订单簿失衡率、诱饵风险系数、期权最大痛点、ATR）
完成以下问题：
1. 做市商最可能穿刺的方向是哪边？穿刺成本如何？
2. 该方向上的大单墙是否构成真实阻力？
3. 期权痛点的磁吸效应是强化还是削弱穿刺方向？
4. 是否存在“穿刺-反杀”的诱饵结构？
5. 清算维度结论：[向上穿刺/向下穿刺/均衡]

【第二步：多空博弈分析】
多头数据确认表（至少5条）：
（列出CVD斜率、主动买卖比、24h期货净流、STH SOPR、现价vs STH成本、恐慌贪婪指数）
空头数据确认表（至少5条）：
（列出OI 24h变化、资金费率分位、大户多空比分位、现货24h净流、交易所BTC余额变化、现货/期货背离度）
多头论据（引用上表）：1. 2. 3.
空头论据（引用上表）：1. 2. 3.
交叉质询：
- 多头攻击空头最脆弱的一环：
- 空头攻击多头最脆弱的一环：
博弈维度结论：[多头优势/空头优势/僵持]

【第三步：预期差分析】
数据确认表：（列出恐慌贪婪指数、合约基差分位、稳定币市值趋势、交易所BTC余额变化、借贷利率、BTC市值占比趋势、P/C比、价格7日分位、波动因子、CGDI分位）
回答：1. 当前价格已定价了什么信息？2. 存在什么未被定价的潜在变化？3. 若价格打破均衡，谁会最意外？
预期差维度结论：[向上预期差/向下预期差/无显著预期差]

【第四步：三维汇聚与信号定性】
汇聚规则：三方一致→重仓/高置信度；两方一致→中仓/中置信度；无一致→观望。
汇聚结果：清算维度映射/博弈维度映射/预期差维度映射/最终方向/仓位/置信度。
信号定性（跨币种修正）：清算方向一致性/CVD方向一致性/定性[系统性趋势/单币种独立行情]/修正后仓位。

【第五步：价格路径推演与合约策略】
关键推演数据确认：最短穿刺方向、清算密集区位置、期权最大痛点、博弈结论、预期差方向、ATR。
价格路径推演：第一段运动…第二段运动…最终目标…
- 币种：{symbol}
- 方向：[做多/做空/观望]
- 仓位：[重仓/中仓/轻仓/无] (依据)
- 置信度：[高/中/低] (依据)
- 入场区间：[__-__] (依据)
- 止损：[__] (依据：设于败方逻辑成立位置之外)
- 止盈：[__] (依据：清算密集区/期权痛点)
- 盈亏比计算：(止盈-入场)/(入场-止损) = __:1
- 说明：[一句话指令或观望触发条件]

【强制输出格式】
{{
  "direction": "做多/做空/观望",
  "confidence": "高/中/低",
  "position_size": "重仓/中仓/轻仓/无",
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "execution_plan": "",
  "reasoning": "完整的五步分析过程，用清晰的标题和换行分隔。",
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
            resp = client.chat.completions.create(model=FAST_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=16384, timeout=TIMEOUT_SECONDS)
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip(): raise ValueError("空响应")
            json_str = extract_json(content)
            s = json.loads(json_str)
            s["direction"] = {"做多":"long","做空":"short","观望":"neutral"}.get(s.get("direction",""), "neutral")
            s["position_size"] = {"重仓":"heavy","中仓":"medium","轻仓":"light","无":"none"}.get(s.get("position_size",""), "none")
            s["confidence"] = {"高":"high","中":"medium","低":"low"}.get(s.get("confidence",""), "medium")
            s.setdefault("reasoning",""); s.setdefault("risk_note",""); s.setdefault("execution_plan","")
            s["reasoning"] = format_reasoning(s["reasoning"])
            s["_model_used"] = resp.model
            return s
        except Exception as e:
            logger.warning(f"交易员调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {"direction":"neutral","confidence":"低","position_size":"none","entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,"execution_plan":"调用失败","reasoning":"调用失败","risk_note":"","_model_used":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

# ------------------- 审计官 -------------------
def call_reviewer(strategy: dict, data: dict, symbol: str) -> dict:
    direction_bias = data.get('direction_bias', 0.0)
    prompt = f"""你是风控审计官。请对交易员策略完成全方位审计，输出严格JSON。

【交易标的】{symbol}【锚点】direction_bias={direction_bias:.3f}
【策略】方向：{strategy.get('direction')}，仓位：{strategy.get('position_size')}
入场：{strategy.get('entry_price_low')}-{strategy.get('entry_price_high')}
止损：{strategy.get('stop_loss')}，止盈：{strategy.get('take_profit')}
【推演】{strategy.get('reasoning', '无')}

【审计要求】
- 按"一、遗漏指标与分析缺失；二、数据与解读错误；三、逻辑错误；四、关键反证提示；五、博弈层面审视"分段。
- 每条发现格式为：在[步骤X]中，交易员[问题]。该指标显示[数值]，若纳入将[强化/削弱/推翻]判断。[严重性：高/中/低]
- 必须统计所有发现的严重性，填入severity对象中。
- 只输出纯JSON。

【输出JSON】
{{
  "verdict": "通过/存疑/驳回",
  "severity": {{"max": "严重/中等/轻度/无", "high": 0, "medium": 0, "low": 0}},
  "full_report": "完整审计文本"
}}
"""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=FAST_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=4096, timeout=120)
            content = resp.choices[0].message.content or ""
            json_str = extract_json(content)
            rev = json.loads(json_str)
            rev["full_report"] = rev.get("full_report", str(rev))
            rev.setdefault("verdict", "通过")
            sev = rev.get("severity", {})
            rev["severity"] = {
                "max": sev.get("max", "无"),
                "high": sev.get("high", 0),
                "medium": sev.get("medium", 0),
                "low": sev.get("low", 0)
            }
            if not rev.get("full_report") or rev["full_report"] == str(rev):
                rev["full_report"] = f"审计完成，结论：{rev['verdict']}，最高严重性：{rev['severity']['max']}。"
            return {**rev, "_model": resp.model}
        except Exception as e:
            logger.warning(f"审计官调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {"verdict":"通过","severity":{"max":"无","high":0,"medium":0,"low":0},"full_report":"审计官调用失败，跳过审计","_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

# ------------------- 交易委员会 -------------------
def call_judge(strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    direction_bias = data.get('direction_bias', 0.0)
    prompt = f"""你是交易委员会主席，拥有最终裁决权。

【标的】{symbol}，现价：{data.get('mark_price', 0):.2f}，锚点：{direction_bias:.3f}

【交易员完整策略】
方向：{strategy.get('direction')}，仓位：{strategy.get('position_size')}，置信度：{strategy.get('confidence')}
入场：{strategy.get('entry_price_low')}-{strategy.get('entry_price_high')}
止损：{strategy.get('stop_loss')}，止盈：{strategy.get('take_profit')}
执行计划：{strategy.get('execution_plan')}
风险提示：{strategy.get('risk_note')}
推演：{strategy.get('reasoning', '无')}

【审计报告】
{reviewer_report.get('full_report', '无审计报告')}
审计结论：{reviewer_report.get('verdict', '未知')}
最高严重性：{reviewer_report.get('severity', {}).get('max', '未知')}

【裁决要求】
- 对于“严重”指控，若成立则必须推翻策略。
- 对于“中等”或“轻度”指控，可选择修改执行或维持原判。
- final_reasoning必须包含对审计指控的逐条回应，这是最终裁决书。
- 维持原判时，价格字段不能填0，必须填写交易员给出的原始数值。
- 推翻后必须给出明确的操作指令（观望触发条件或新策略）。
- 只输出纯JSON。

【输出JSON】
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
  "final_reasoning": "裁决书正文，必须包含对审计指控的逐条回应"
}}
"""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=REASONING_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=16384, timeout=120)
            content = resp.choices[0].message.content or ""
            json_str = extract_json(content)
            result = json.loads(json_str)
            result["final_direction"] = {"做多":"long","做空":"short","观望":"neutral"}.get(result.get("final_direction",""), "neutral")
            result["final_position_size"] = {"重仓":"heavy","中仓":"medium","轻仓":"light","无":"none"}.get(result.get("final_position_size",""), "none")
            result["final_confidence"] = {"高":"high","中":"medium","低":"low"}.get(result.get("final_confidence",""), "medium")
            if result.get("final_verdict") == "维持原判":
                result["entry_price_low"] = result.get("entry_price_low") or strategy.get("entry_price_low", 0)
                result["entry_price_high"] = result.get("entry_price_high") or strategy.get("entry_price_high", 0)
                result["stop_loss"] = result.get("stop_loss") or strategy.get("stop_loss", 0)
                result["take_profit"] = result.get("take_profit") or strategy.get("take_profit", 0)
                result["execution_plan"] = result.get("execution_plan") or strategy.get("execution_plan", "")
                result["risk_note"] = result.get("risk_note") or strategy.get("risk_note", "")
            if result.get("final_verdict") == "推翻" and result.get("final_direction") == "neutral":
                if not result.get("execution_plan"): result["execution_plan"] = "当前策略逻辑崩塌，等待新的三维一致信号。"
                result["entry_price_low"] = 0; result["entry_price_high"] = 0
                result["stop_loss"] = 0; result["take_profit"] = 0
            result["final_reasoning"] = result.get("final_reasoning") or "裁决完成，详见裁决书。"
            return {**result, "_model": resp.model}
        except Exception as e:
            logger.warning(f"委员会调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {"final_verdict":"维持原判","final_direction":strategy.get("direction","neutral"),"final_confidence":strategy.get("confidence","低"),"final_position_size":strategy.get("position_size","none"),"entry_price_low":strategy.get("entry_price_low",0),"entry_price_high":strategy.get("entry_price_high",0),"stop_loss":strategy.get("stop_loss",0),"take_profit":strategy.get("take_profit",0),"execution_plan":strategy.get("execution_plan",""),"risk_note":strategy.get("risk_note",""),"audit_adopted":False,"audit_max_severity":"无","final_reasoning":"委员会调用失败，自动维持原判","_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

def apply_final_verdict(strategy: dict, judge_result: dict) -> dict:
    verdict = judge_result.get("final_verdict", "维持原判")
    logger.info(f"应用最终决议: {verdict}")
    strategy["_judge_verdict"] = verdict
    strategy["_judge_reasoning"] = judge_result.get("final_reasoning", "")
    if verdict == "推翻":
        if judge_result.get("final_direction") == "neutral":
            _force_neutral(strategy, "委员会推翻并改为观望")
        else:
            for k in ["direction","confidence","position_size","entry_price_low","entry_price_high","stop_loss","take_profit","execution_plan"]:
                if k in judge_result: strategy[k] = judge_result[k]
        strategy["risk_note"] = judge_result.get("risk_note", strategy.get("risk_note", ""))
        return strategy
    elif verdict == "修改执行":
        for k in ["confidence","position_size","entry_price_low","entry_price_high","stop_loss","take_profit","execution_plan"]:
            if judge_result.get(k): strategy[k] = judge_result[k]
        strategy["risk_note"] = judge_result.get("risk_note", strategy.get("risk_note", ""))
        return strategy
    else:
        strategy["risk_note"] = judge_result.get("risk_note") or strategy.get("risk_note", "")
        strategy["execution_plan"] = judge_result.get("execution_plan") or strategy.get("execution_plan", "")
        return strategy