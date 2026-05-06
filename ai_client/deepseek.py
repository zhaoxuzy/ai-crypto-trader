import os, json, time, re, math
from datetime import datetime
from openai import OpenAI
from utils.logger import logger

FAST_MODEL = "deepseek-v4-pro"
REASONING_MODEL = "deepseek-v4-pro"
MAX_RETRIES = 3
RETRY_BASE_WAIT = 2
TIMEOUT_SECONDS = 180

VALID_DIRECTIONS = {"long", "short", "neutral"}
VALID_CONFIDENCES = {"high", "medium", "low"}
VALID_POSITION_SIZES = {"heavy", "medium", "light", "none"}

# ---------- 辅助函数 ----------
def norm_direction(raw: str) -> str:
    if not raw: return "neutral"
    clean = raw.strip().lower()
    if clean in VALID_DIRECTIONS: return clean
    mapping = {"做多": "long", "做空": "short", "观望": "neutral"}
    return mapping.get(clean, "neutral")

def norm_confidence(raw: str) -> str:
    if not raw: return "medium"
    clean = raw.strip().lower()
    if clean in VALID_CONFIDENCES: return clean
    mapping = {"高": "high", "中": "medium", "低": "low"}
    return mapping.get(clean, "medium")

def norm_position_size(raw: str) -> str:
    if not raw: return "none"
    clean = raw.strip().lower()
    if clean in VALID_POSITION_SIZES: return clean
    mapping = {"重仓": "heavy", "中仓": "medium", "轻仓": "light", "无": "none", "none": "none"}
    return mapping.get(clean, "none")

def format_reasoning(text: str) -> str:
    if not text: return text
    text = text.replace('\\n', '\n')
    text = re.sub(r'(\*\*[^*]+\*\*)', r'\n\1\n', text)
    text = re.sub(r'(【[^】]+】)', r'\n\1\n', text)
    text = re.sub(r'(第[一二三四五六七八九十]+步[：:])', r'\n\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _log_response(role: str, prompt: str, content: str, reasoning: str = None):
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/{role}_{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "content": content, "reasoning": reasoning}, f, ensure_ascii=False, indent=2)
    except: pass

def extract_json_safe(content: str) -> str:
    if not content or not content.strip():
        raise ValueError("空响应内容")
    m = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if m:
        json_str = m.group(1).strip()
        try: json.loads(json_str); return json_str
        except: pass
    m = re.search(r'```\s*([\s\S]*?)\s*```', content)
    if m:
        json_str = m.group(1).strip()
        try: json.loads(json_str); return json_str
        except: pass
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = content[start:end+1].strip()
        try: json.loads(json_str); return json_str
        except: pass
    if start != -1 and end != -1:
        json_str = content[start:end+1].strip()
        json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        try: json.loads(json_str); logger.warning("JSON通过转义修复成功"); return json_str
        except: pass
    if json_str.startswith('{'):
        if not json_str.endswith('"') and not json_str.endswith('}'):
            json_str += '"}'
        elif json_str.endswith('"'):
            json_str += '}'
        try: json.loads(json_str); logger.warning("JSON通过暴力修补修复成功"); return json_str
        except: pass
    raise ValueError(f"JSON提取失败，前200字符: {content[:200]}")

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
        atr_15m = data.get("atr_15m",0)
        mark = data.get("mark_price",0)
        ab_liq = data.get("above_liq",0)
        bl_liq = data.get("below_liq",0)
        bias_q = data.get("_bias_quality","reliable")
        bias = data.get("direction_bias",0.0)
        if (not ab_liq or ab_liq<=0) and (not bl_liq or bl_liq<=0) and direction!="neutral":
            _force_neutral(s, "清算数据缺失"); return True, ""
        if atr_15m<=0 or mark<=0:
            if direction!="neutral": _force_neutral(s, "ATR或价格缺失"); return True, ""
        if bias_q in ("reliable","degraded") and abs(bias)>0.4 and direction!="neutral":
            if (bias>0 and direction=="short") or (bias<0 and direction=="long"):
                _force_neutral(s, f"方向与锚点({bias:.3f})冲突"); return True, ""
    if direction=="neutral":
        for f in ["entry_price_low","entry_price_high","stop_loss","take_profit"]: s[f]=0
        s["position_size"]="none"
        if not s.get("execution_plan"): s["execution_plan"]="等待触发条件"
        return True, ""
    for f in ["entry_price_low","entry_price_high","stop_loss","take_profit"]:
        val = s.get(f)
        if val is None or float(val)<=0: return False, f"缺少或无效的 {f}"
    return True, ""

# ---------- 清算穿刺 & 仪表盘 ----------
def compute_liquidation_bias(data: dict) -> dict:
    liq_r = data.get('liq_ratio',1.0)
    cvd = data.get('cvd_slope',0.0)
    taker = data.get('taker_ratio_1h',0.5)
    ob_imb = data.get('orderbook_imbalance',0.0)
    press = data.get('large_order_pressure',0.0)
    pain = data.get('max_pain',0.0)
    atr = data.get('atr',0.0)
    mark = data.get('mark_price',0.0)
    score = (liq_r-1.0)*0.4 + (1 if cvd>0 else -1)*0.3 + (taker-0.5)*0.3
    direction = 'balanced'
    if score>0.15: direction='up'
    elif score<-0.15: direction='down'
    lure = (direction=='up' and press<-0.5) or (direction=='down' and press>0.5)
    pain_eff = False
    if atr>0 and pain>0 and abs(pain-mark)<1.0*atr:
        if (direction=='up' and pain>mark) or (direction=='down' and pain<mark): pain_eff=True
    return {'puncture_direction':direction,'puncture_score':score,'lure_risk':lure,'pain_magnet':pain_eff}

def build_expectation_dashboard(data: dict) -> str:
    basis_ann = data.get('basis_annualized',0)
    basis_med = data.get('basis_median',8)
    fund_pct = data.get('funding_percentile',50)
    cgdi_pct = data.get('cgdi_percentile',50)
    st_flow = data.get('stablecoin_trend_7d',0)
    btc_dom = data.get('btc_dominance_trend_7d',0)
    borrow = data.get('borrow_rate',0)*100
    pc = data.get('put_call_ratio',1.0)
    price_pct = data.get('price_percentile',50)
    vol_f = data.get('vol_factor',1.0)
    return f"""| 指标 | 当前值 | 历史基线 | 定价了什么？ |
|------|--------|----------|------------|
| 3月基差年化 | {basis_ann:.1f}% | {basis_med:.1f}% | >基线时期货溢价过热 |
| 资金费率分位 | {fund_pct:.0f}% | 50% | 多头支付意愿 |
| CGDI分位 | {cgdi_pct:.0f}% | 50% | 综合贪婪度 |
| 稳定币净流7d | {st_flow:+.1f}% | +0.5% | 资金面松紧 |
| BTC.D趋势7d | {btc_dom:+.1f}% | 0% | 风险偏好 |
| 借贷利率 | {borrow:.2f}% | 均值 | 杠杆紧张度 |
| P/C比 | {pc:.3f} | 0.7 | >1恐慌对冲 |
| 价格7日分位 | {price_pct:.0f}% | 50% | 超买/超卖 |
| 波动因子 | {vol_f:.2f} | 1.0 | 不确定性定价 |"""

CORE_KEYS = [
    'mark_price','atr','above_liq','below_liq','liq_ratio',
    'cvd_slope','taker_ratio_1h','oi_change_24h','funding_percentile',
    'orderbook_imbalance','large_order_pressure','max_pain','put_call_ratio',
    'basis_percentile','stablecoin_trend_7d','cgdi_percentile',
    'fear_greed','lth_realized_price','sth_realized_price','sth_sopr'
]

def compute_coverage(data: dict) -> dict:
    total = len(CORE_KEYS)
    available = sum(1 for k in CORE_KEYS if data.get(k) is not None)
    coverage = available / total if total > 0 else 0.0
    return {"available": available, "total": total, "coverage": coverage}

# ------------------- 新七步 Prompt -------------------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    if cross_symbol is None:
        cross_symbol = "ETH" if symbol=="BTC" else "BTC"
    coverage = compute_coverage(data)

    def safe_val(key, default=0.0, scale=1.0, fmt=".2f"):
        raw = data.get(key)
        if raw is None: return ("[N/A]", True)
        try: val = float(raw)*scale
        except: return ("[N/A]", True)
        if math.isnan(val) or math.isinf(val): return ("[N/A]", True)
        try: return (f"{val:{fmt}}", False)
        except: return ("[N/A]", True)

    # 常用字段
    mark_str, _ = safe_val('mark_price', fmt=".2f")
    atr_str, _ = safe_val('atr', fmt=".2f")
    fear_greed = data.get('fear_greed',50)
    lth_str, _ = safe_val('lth_realized_price', fmt=".2f")
    sth_str, _ = safe_val('sth_realized_price', fmt=".2f")
    sopr_str, _ = safe_val('sth_sopr',1.0,fmt=".3f")
    stable_str, _ = safe_val('stablecoin_trend_7d',fmt="+.1f")
    oi_chg_str, _ = safe_val('oi_change_24h',fmt="+.1f")
    fund_pct_str, _ = safe_val('funding_percentile',50,fmt=".0f")
    cvd_str, _ = safe_val('cvd_slope',fmt=".4f")
    taker_str, _ = safe_val('taker_ratio_1h',fmt=".3f")
    nf24h_str, _ = safe_val('netflow_24h',scale=1/1e6,fmt=".1f")
    abv_liq_str, _ = safe_val('above_liq',scale=1/1e9,fmt=".2f")
    blw_liq_str, _ = safe_val('below_liq',scale=1/1e9,fmt=".2f")
    liq_r_str, _ = safe_val('liq_ratio',fmt=".2f")
    abv_trig = data.get('above_trigger','N/A')
    blw_trig = data.get('below_trigger','N/A')
    lgs_str, _ = safe_val('large_sell_value',scale=1/1e6,fmt=".1f")
    lgb_str, _ = safe_val('large_buy_value',scale=1/1e6,fmt=".1f")
    press_str, _ = safe_val('large_order_pressure',fmt=".3f")
    ob_imb_str, _ = safe_val('orderbook_imbalance',fmt=".3f")
    lure_str, _ = safe_val('lure_risk_factor',fmt=".2f")
    pain_str, _ = safe_val('max_pain',fmt=".2f")
    pc_str, _ = safe_val('put_call_ratio',fmt=".4f")
    basis_pct_str, _ = safe_val('basis_percentile',50,fmt=".0f")
    btc_dom_str, _ = safe_val('btc_dominance_trend_7d',fmt="+.1f")
    borrow_str, _ = safe_val('borrow_rate',scale=100,fmt=".2f")
    exch_str, _ = safe_val('exchange_btc_change_24h',fmt="+.0f")
    spot24_str, _ = safe_val('spot_netflow_24h',scale=1/1e6,fmt=".1f")
    spot_div_str, _ = safe_val('spot_vs_futures_divergence',fmt=".2f")
    top_ls_str, _ = safe_val('top_ls_percentile',50,fmt=".0f")
    price_pct_str, _ = safe_val('price_percentile',50,fmt=".0f")
    vol_f_str, _ = safe_val('vol_factor',1.0,fmt=".2f")
    cgdi_pct_str, _ = safe_val('cgdi_percentile',50,fmt=".0f")
    direction_bias = data.get('direction_bias',0.0)
    bias_quality = data.get('_bias_quality','reliable')

    puncture = compute_liquidation_bias(data)
    dashboard = build_expectation_dashboard(data)

    # 跨币种上下文
    cross_context = ""
    if eth_data:
        cross_context = f"""
【跨币种数据（{cross_symbol}）】
| 指标 | {cross_symbol} | {symbol} |
|------|----------------|----------|
| 清算比值 | {eth_data.get('liq_ratio',0):.2f} | {data.get('liq_ratio',1):.2f} |
| CVD斜率 | {eth_data.get('cvd_slope',0):.4f} | {data.get('cvd_slope',0):.4f} |
| OI 24h变化 | {eth_data.get('oi_change_24h',0):+.1f}% | {data.get('oi_change_24h',0):+.1f}% |
| 顶多空分位 | {eth_data.get('top_ls_percentile',50):.0f}% | {data.get('top_ls_percentile',50):.0f}% |
| 爆仓偏空比 | {eth_data.get('liq_bias_1h',0):.3f} | {data.get('liq_bias_1h',0):.3f} |
| 资金费率分位 | {eth_data.get('funding_percentile',50):.0f}% | {fund_pct_str}% |
| 期权痛点 | {eth_data.get('max_pain',0):.2f} | {pain_str} |
规则：若两币方向一致→系统性趋势；矛盾→单币种行情，仓位降一级。"""
    else:
        cross_context = "【跨币种数据不可用】仓位上限下调一级，置信度上限为'中'。"

    # 未使用指标提示
    unused_list = [
        "retail_whale_divergence", "cvd_acceleration", "oi_acceleration",
        "spot_vs_futures_divergence", "exchange_btc_change_24h",
        "lth_realized_price", "sth_realized_price", "lth_sopr", "sth_sopr",
        "cgdi_current", "stablecoin_mcap", "eth_btc_percentile",
        "cross_funding_percentile", "cross_max_pain"
    ]
    unused_note = "\n".join(f"- {x}" for x in unused_list)

    prompt =prompt = f"""你是一位拥有 15 年实战经验、以量化严谨著称的加密货币首席交易员。你的分析风格是冷酷、计算式和多情景推演，你必须严格按照「七步递进分析框架」完成推演。每一步都必须展示你的具体“计算”或“因果构建”过程，而不是仅仅罗列指标名称。总推理字数 ≤ 3000 字，以纯 JSON 格式输出。

【数据与锚点】
覆盖率：{coverage['coverage']:.0%}（{coverage['available']}/{coverage['total']}）
方向锚点 direction_bias = {direction_bias:.3f}，可信度：{bias_quality}
清算穿刺预判方向：{puncture['puncture_direction']}，得分：{puncture['puncture_score']:.2f}
诱饵风险：{puncture['lure_risk']}，期权磁吸：{puncture['pain_magnet']}

{ dashboard }

【市场数据】
现价：{mark_str}，ATR：{atr_str}，恐慌贪婪：{fear_greed}
LTH成本：{lth_str}，STH成本：{sth_str}，STH SOPR：{sopr_str}
稳定币趋势7d：{stable_str}%，BTC.D趋势7d：{btc_dom_str}%
交易所BTC变化24h：{exch_str} BTC，借贷利率：{borrow_str}%
CGDI绝对值：{data.get('cgdi_current',0):.0f}，稳定币市值：{data.get('stablecoin_mcap',0)/1e9:.1f}B

【清算与动能】
上方清算：{abv_liq_str}B，最近簇距{abv_trig}点，下方清算：{blw_liq_str}B，距{blw_trig}，比值：{liq_r_str}
期权痛点：{pain_str}，P/C比：{pc_str}
CVD斜率：{cvd_str}，加速度：{data.get('cvd_acceleration',0):.4f}
主动买卖比(1h)：{taker_str}，大单压迫比：{press_str} (买{lgb_str}M/卖{lgs_str}M)
OI 24h变化：{oi_chg_str}%，OI加速度：{data.get('oi_acceleration',0):.4f}
订单簿失衡：{ob_imb_str}，诱饵因子：{lure_str}
散户/鲸鱼背离：{data.get('retail_whale_divergence',0):.3f}，全球多空比：{data.get('global_ls_ratio',1):.2f}
爆仓偏空比(1h)：{data.get('liq_bias_1h',0):.3f}，多爆：{data.get('long_liq_1h',0):.2f}M，空爆：{data.get('short_liq_1h',0):.2f}M
期现背离：{spot_div_str} (期货24h净流{nf24h_str}M，现货{spot24_str}M)
价格7日分位：{price_pct_str}%，波动因子：{vol_f_str}，CGDI分位：{cgdi_pct_str}%

{ cross_context }

【可参考的附加指标（思维扩展，选择其中至少一项融入分析）】
{unused_note}

---
# 七步分析框架（严格按此顺序，且每一步都必须包含具体的推导动作）

## 步骤1：数据全景与可信度评估
- **动作**：识别覆盖率缺口，判断`bias_quality`。如果覆盖率<70%或锚点可信度低，必须在后续 `confidence` 和 `risk_note` 中强制降级。
- **输出**：明确的置信度上限声明（例如：“本报告最大置信度不超过70%”）。

## 步骤2：宏观结构与链上底色
- **强制动作**：你必须用下面这个因果模板来构建你的宏观判断：“当前市场处于【牛/熊/震荡】的【早期/中期/晚期】，核心证据是【LTH/STH成本与现价关系】揭示了【持股者状态】，结合【稳定币/交易所存量】的变动，表明资金是【流入/流出】。因此，结构性支撑带在【LTH成本或高成交量节点】，宏观偏向为【偏多/偏空/中性】。”
- **输出**：宏观象限、结构性支撑/压力带、宏观偏向。

## 步骤3：多空动能正交分解
- **强制计算**：你必须为以下每个因子进行标准化打分（范围-2到+2，-2为强烈看空，+2为强烈看多），并在一句话内简述理由。
  【上方清算簇、CVD斜率/加速度、主动买卖比、大单压迫比、OI加速度、订单簿失衡】
- **强制计算**：然后结合【散户/鲸鱼背离】和【全球多空比】进行±0.5分的情景修正。
- **输出**：“标准化多空净得分 = (多头因子得分总和 + 空头因子得分总和) / 因子数，净倾向为【强多/偏多/中性/偏空/强空】”。

## 步骤4：流动性猎杀博弈
- **强制推演**：你必须同时推演上方和下方两种猎杀路径。
  - **路径A（向【上/下】）**：“价格向【上/下】移动1-2 ATR（即到【X价格】），将引爆【Y金额】的空/多单，这笔燃料足以将价格进一步推向【Z目标】，且这与【期权痛点/磁吸】形成共振。”
  - **路径B（向【下/上】）**：“反之，若价格先向【下/上】假突破至【W价格】，则会触发【诱饵风险】，清扫早期跟风盘，并可能形成更健康的【上行/下行】基础。”
- **结论**：经过风险收益比权衡，当前最可能的猎杀方向及理由。

## 步骤5：预期差仪表盘解读
- **强制寻找矛盾**：你必须找出至少两个相互矛盾的指标对（例如：“恐慌贪婪指数显示‘极度贪婪’，但【稳定币净流出/BTC交易所存量增加/CVD负值】却显示聪明钱在撤退”）。
-  **推演意外**：“基于这个矛盾，当价格向【市场情绪指向的相反方向】移动时，谁会感到最意外并被迫平仓？”

## 步骤6：跨币种生态验证
- **强制对比**：必须对比至少一个主流币（ETH）的清算图、CVD、资金费率。判断这是BTC的独立行情，还是有生态协同的系统性趋势。
- **输出**：明确的仓位修正建议（例如：“若此时做多BTC，应规避做多ETH，因其汇率分位和清算结构更弱”）。

## 步骤7：策略生成与反向压力测试
- **综合加权**：严格按宏观20%、动能25%、博弈30%、预期差15%、跨币种10%的权重，得出最终方向。
- **反向压力测试（强制构建第二故事线）**：“现在，请你构建一个与上述主力逻辑完全相反的、逻辑自洽的故事线。它必须：
  1. 找到当前数据中一个支持此反向逻辑的‘种子信号’（例如一个异常的链上转移或微小的订单簿异动）。
  2. 描述这个种子信号如何演变，最终推翻当前的主力逻辑。
  3. 给出一个明确的条件，当此条件触发时，你应该立即退出主力仓位。”
- **输出**：入场价格区间、止损价、止盈价、盈亏比、具体执行计划。

【输出JSON格式，reasoning字段中每步以【步骤X：名称】开头】
{{
  “direction“: “做多/做空/观望”,
  “confidence”: “高/中/低”,
  “position_size”: “重仓/中仓/轻仓/无”,
  “entry_price_low”: 0.0,
  “entry_price_high”: 0.0,
  “stop_loss”: 0.0,
  “take_profit”: 0.0,
  “execution_plan”: “详细的订单执行策略，如分步挂单”,
  “reasoning”: “严格七步分析，每一步都包含强制性的计算或因果链”、
  “risk_note”: “必须包含基于步骤1可信度和步骤7反向情景的警告”、
  “risk_reward_ratio”: 0.0,
  “vote_result”: {{“宏观底色”: “”, “多空动能”: “”, “博弈维度”: “”, “预期差”: “”, “跨币种”: “”, “最终方向”: “”}}
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
                messages=[{"role":"user","content":prompt}],
                max_tokens=16384,
                timeout=TIMEOUT_SECONDS,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content or ""
            _log_response("trader", prompt, content)
            if not content.strip(): raise ValueError("空响应")
            json_str = extract_json_safe(content)
            s = json.loads(json_str)
            s["direction"] = norm_direction(s.get("direction",""))
            s["position_size"] = norm_position_size(s.get("position_size",""))
            s["confidence"] = norm_confidence(s.get("confidence",""))
            s.setdefault("reasoning",""); s.setdefault("risk_note",""); s.setdefault("execution_plan","")
            s["reasoning"] = format_reasoning(s["reasoning"])
            s["_model_used"] = resp.model
            return s
        except Exception as e:
            logger.warning(f"交易员调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {"direction":"neutral","confidence":"low","position_size":"none","entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,"execution_plan":"调用失败","reasoning":"调用失败","risk_note":"","_model_used":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

# ------------------- 审计官（新版七步审计） -------------------
def call_reviewer(strategy: dict, data: dict, symbol: str) -> dict:
    direction_bias = data.get('direction_bias',0.0)
    coverage_info = compute_coverage(data)
    puncture = compute_liquidation_bias(data)
    bias_quality = data.get('_bias_quality','reliable')

    prompt = f"""你是一位独立的风险审计官，对首席交易员的七步分析进行逐步骤审查。
你必须输出七步审计表，每步给出结论与问题，严重程度区分。未使用的附加指标若存在重大矛盾必须提及。

【审计背景】
标的：{symbol}，锚点：{direction_bias:.3f}，锚点可信度：{bias_quality}
覆盖率：{coverage_info['coverage']:.0%}，穿刺预判：{puncture['puncture_direction']}
交易员方向：{strategy.get('direction')}，仓位：{strategy.get('position_size')}

交易员推演原文：
{format_reasoning(strategy.get('reasoning','无'))}

请按以下JSON输出，务必完整：
{{
  "step_audits": [
    {{
      "step": 1,
      "verdict": "合格/存在瑕疵/严重错误",
      "issues": [
        {{"type": "数据遗漏/误读/逻辑矛盾/反证缺失", "description": "...", "severity": "高/中/低", "evidence": "..."}}
      ]
    }},
    ... (步骤2至7)
  ],
  "overall_verdict": "通过/存疑/驳回",
  "max_severity": "严重/中等/轻度/无",
  "severity_summary": {{"严重":0,"中等":0,"轻度":0}},
  "full_report": "文字报告，特别是被忽略的附加指标中是否有重大矛盾"
}}
"""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=120)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=8192,
                timeout=120,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content or ""
            _log_response("reviewer", prompt, content)
            rev = json.loads(extract_json_safe(content))
            if "full_report" not in rev or not rev["full_report"].strip():
                rev["full_report"] = content
            rev["full_report"] = format_reasoning(rev["full_report"])
            if sum(rev.get("severity_summary",{}).values())==0 and rev.get("overall_verdict")=="驳回":
                rev["severity_summary"]["严重"] = 1
                rev["max_severity"] = "严重"
            return {**rev, "_model": resp.model}
        except Exception as e:
            logger.warning(f"审计官调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {"overall_verdict":"驳回","max_severity":"严重","severity_summary":{"严重":1,"中等":0,"轻度":0},"step_audits":[],"full_report":"审计失败","_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

# ------------------- 交易委员会（新版仲裁） -------------------
def call_judge(strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    direction_bias = data.get('direction_bias',0.0)

    # 提取审计指控
    audit_charges = ""
    for audit in reviewer_report.get("step_audits",[]):
        for issue in audit.get("issues",[]):
            if issue.get("severity") in ("高","中"):
                audit_charges += f"步骤{audit['step']}: {issue['description']} (严重性:{issue['severity']})\n"

    prompt = f"""你是交易委员会主席，拥有最终决策权。你必须逐条回应审计官的严重/中等指控，并重新加权七步信号输出最终策略。

【标的】{symbol}，现价：{data.get('mark_price',0):.2f}，锚点：{direction_bias:.3f}
交易员原策略：方向{strategy.get('direction')}，仓位{strategy.get('position_size')}
审计结论：{reviewer_report.get('overall_verdict')}，最高严重性：{reviewer_report.get('max_severity')}

审计指控：
{audit_charges if audit_charges else "无严重指控"}

请输出以下JSON，包含对各指控的回应及加权信号：
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
  "audit_responses": [
    {{"step":2,"issue":"...","adopted":true,"reason":"..."}}
  ],
  "weighted_signal": {{
    "step2": "bearish/bullish/neutral",
    "step3": "...",
    "step4": "...",
    "step5": "...",
    "step6": "...",
    "composite_score": -0.5
  }},
  "final_reasoning": "裁决理由，包括加权过程"
}}
"""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=120)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=REASONING_MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=16384,
                timeout=120,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content or ""
            _log_response("judge", prompt, content)
            result = json.loads(extract_json_safe(content))
            result["final_direction"] = norm_direction(result.get("final_direction",""))
            result["final_position_size"] = norm_position_size(result.get("final_position_size",""))
            result["final_confidence"] = norm_confidence(result.get("final_confidence",""))
            if result.get("final_verdict")=="维持原判":
                for f in ["entry_price_low","entry_price_high","stop_loss","take_profit","execution_plan","risk_note"]:
                    if not result.get(f): result[f] = strategy.get(f,0)
            result["final_reasoning"] = format_reasoning(result.get("final_reasoning",""))
            return {**result, "_model": resp.model}
        except Exception as e:
            logger.warning(f"委员会调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {"final_verdict":"推翻","final_direction":"neutral","final_confidence":"low","final_position_size":"none","entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,"execution_plan":"失败","risk_note":"","final_reasoning":"失败","_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

def apply_final_verdict(strategy: dict, judge_result: dict) -> dict:
    verdict = judge_result.get("final_verdict","维持原判")
    logger.info(f"应用最终决议: {verdict}")
    strategy["_judge_verdict"] = verdict
    strategy["_judge_reasoning"] = judge_result.get("final_reasoning","")
    fields = ["direction","confidence","position_size","entry_price_low","entry_price_high","stop_loss","take_profit","execution_plan","risk_note"]
    if verdict in ("推翻","修改执行"):
        if judge_result.get("final_direction") == "neutral":
            _force_neutral(strategy, "委员会改为观望")
        else:
            for k in fields:
                if k in judge_result and judge_result[k] is not None:
                    strategy[k] = judge_result[k]
    return strategy
