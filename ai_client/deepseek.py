"""
deepseek.py — 生产级三角色闭环 (启用JSON强制模式 + 保留容错)
- 所有模型调用均启用 response_format={"type": "json_object"}
- 多级JSON提取与修复策略，确保高可用
- 取消微观数据时效约束
- 完整闭环：交易员 -> 审计官 -> 委员会
"""

import os, json, time, re, math
from datetime import datetime
from openai import OpenAI
from utils.logger import logger

TICK_SIZE = 0.1
MAX_RETRIES = 3
RETRY_BASE_WAIT = 2
TIMEOUT_SECONDS = 180

FAST_MODEL = "deepseek-v4-pro"
REASONING_MODEL = "deepseek-v4-pro"

VALID_DIRECTIONS = {"long", "short", "neutral"}
VALID_CONFIDENCES = {"high", "medium", "low"}
VALID_POSITION_SIZES = {"heavy", "medium", "light", "none"}

# ---------- 标准化映射 ----------
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

# ---------- 文本格式化 ----------
def format_reasoning(text: str) -> str:
    if not text: return text
    text = text.replace('\\n', '\n')
    text = re.sub(r'(\*\*[^*]+\*\*)', r'\n\1\n', text)
    text = re.sub(r'(【[^】]+】)', r'\n\1\n', text)
    text = re.sub(r'(第[一二三四五六七八九十]+步[：:])', r'\n\1', text)
    text = re.sub(r'(价格路径推演[：:])', r'\n\1', text)
    text = re.sub(r'(多头论据|空头论据|交叉质询|博弈维度结论)', r'\n\1', text)
    text = re.sub(r'(清算维度结论|最终合约策略|入场区间|止损|止盈)', r'\n\1', text)
    text = re.sub(r'(?<=[。！？；：])\s*', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ---------- 辅助 ----------
def _log_response(role: str, prompt: str, content: str, reasoning: str = None):
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/{role}_{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "content": content, "reasoning": reasoning}, f, ensure_ascii=False, indent=2)
    except: pass

def extract_json_safe(content: str) -> str:
    """
    多级降级策略提取JSON，自动修复常见格式错误
    """
    if not content or not content.strip():
        raise ValueError("空响应内容")

    # 第1级：尝试从```json代码块中提取
    m = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if m:
        json_str = m.group(1).strip()
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass

    # 第2级：尝试从```代码块中提取
    m = re.search(r'```\s*([\s\S]*?)\s*```', content)
    if m:
        json_str = m.group(1).strip()
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass

    # 第3级：从整个文本中寻找第一个{到最后一个}之间的内容
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = content[start:end+1].strip()
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass

    # 第4级：修复常见JSON错误 - 转义未处理的特殊字符
    if start != -1 and end != -1:
        json_str = content[start:end+1].strip()
    else:
        json_str = content.strip()
    # 处理字符串中未转义的控制字符
    json_str = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\', json_str)  # 修复单个反斜杠
    json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    try:
        json.loads(json_str)
        logger.warning("JSON通过转义修复成功")
        return json_str
    except json.JSONDecodeError:
        pass

    # 第5级：暴力修补 - 在结尾补上缺失的引号和括号
    if json_str.startswith('{'):
        if not json_str.endswith('"') and not json_str.endswith('}'):
            json_str += '"}'
        elif json_str.endswith('"'):
            json_str += '}'
        try:
            json.loads(json_str)
            logger.warning("JSON通过暴力修补修复成功")
            return json_str
        except json.JSONDecodeError:
            pass

    raise ValueError(f"所有JSON提取策略均失败，原始内容前200字符: {content[:200]}")

def try_parse_reviewer_json(content: str) -> dict:
    """专门用于解析审计官输出的容错函数"""
    # 尝试直接解析为JSON
    try:
        json_str = extract_json_safe(content)
        return json.loads(json_str)
    except Exception:
        pass

    # 从文本中提取关键字段作为fallback
    result = {"verdict": "存疑", "max_severity": "中等", "severity_counts": {"严重": 0, "中等": 1, "轻度": 0}, "full_report": content}

    # 尝试提取verdict
    if "驳回" in content:
        result["verdict"] = "驳回"
        result["max_severity"] = "严重"
        result["severity_counts"]["严重"] = 1
    elif "通过" in content and "驳回" not in content and "存疑" not in content:
        result["verdict"] = "通过"
        result["max_severity"] = "无"
        result["severity_counts"]["中等"] = 0

    # 统计严重性标记
    cnt = {"严重": 0, "中等": 0, "轻度": 0}
    for line in content.split('\n'):
        if '严重性：高' in line: cnt["严重"] += 1
        elif '严重性：中' in line: cnt["中等"] += 1
        elif '严重性：低' in line: cnt["轻度"] += 1
    if sum(cnt.values()) > 0:
        result["severity_counts"] = cnt
        if cnt["严重"] > 0:
            result["verdict"] = "驳回"
            result["max_severity"] = "严重"
        elif cnt["中等"] > 0:
            result["max_severity"] = "中等"
        elif cnt["轻度"] > 0:
            result["max_severity"] = "轻度"

    logger.warning("使用fallback方式解析审计官响应")
    return result

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

# ---------- 清算穿刺 ----------
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

# ---------- 预期定价仪表盘 ----------
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
    return f"""【预期定价仪表盘】
| 指标 | 当前值 | 历史基线 | 定价了什么？ |
|------|--------|----------|------------|
| 3月基差年化 | {basis_ann:.1f}% | {basis_med:.1f}% | >基线时期货溢价过热 |
| 资金费率分位 | {fund_pct:.0f}% | 50% | 多头支付意愿 |
| CGDI分位 | {cgdi_pct:.0f}% | 50% | 综合贪婪度 |
| 稳定币净流7d | {st_flow:+.1f}% | +0.5% | 资金面松紧 |
| BTC.D趋势7d | {btc_dom:+.1f}% | 0% | 风险偏好 |
| 借贷利率 | {borrow:.2f}% | 均值 | 杠杆紧张度 |
| P/C比 | {pc:.3f} | 0.7 | >1恐慌对冲 |
| 价格7日分位 | {price_pct:.0f}% | 50% | 超买/超卖 |
| 波动因子 | {vol_f:.2f} | 1.0 | 不确定性定价 |

预期差分析必须回答：
1. 市场定价最极端的方向（贪婪或恐惧）是什么？依据仪表盘哪些指标？
2. 找出与极端定价矛盾的两个指标，构成潜在“预期差”。
3. 若价格朝矛盾方向移动1 ATR，谁会最意外？
4. 结论：预期差方向必须基于矛盾证据，不可猜测，且必须与清算预判、多空博弈结论交叉核对。"""

# ---------- 核心数据覆盖率 ----------
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

# ------------------- 首席交易员提示词 -------------------
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

    cross_context = ""
    if eth_data:
        cross_context = f"""
【跨币种数据（{cross_symbol}）—— 用于第二步与第四步】
| 指标 | {cross_symbol} 当前值 | {symbol} 当前值 |
|------|---------------------|-----------------|
| 清算比值 | {eth_data.get('liq_ratio',0):.2f} | {data.get('liq_ratio',1):.2f} |
| CVD斜率 | {eth_data.get('cvd_slope',0):.4f} | {data.get('cvd_slope',0):.4f} |
| OI 24h变化 | {eth_data.get('oi_change_24h',0):+.1f}% | {data.get('oi_change_24h',0):+.1f}% |
| 顶多空分位 | {eth_data.get('top_ls_percentile',50):.0f}% | {data.get('top_ls_percentile',50):.0f}% |
| 爆仓偏空比 | {eth_data.get('liq_bias_1h',0):.3f} | {data.get('liq_bias_1h',0):.3f} |
规则：第二步多空论据可引用，交叉质询矛盾信号须作为攻击依据。
第四步信号定性：若清算方向一致→系统性趋势，仓位不变；矛盾→单币种独立行情，仓位降一级。"""
    else:
        cross_context = """
【跨币种数据不可用】
本轮策略的仓位上限自动下调一级，置信度上限为'中'。"""

    core_missing = [k for k in ["kline","heatmap","cvd"] if data.get("data_quality",{}).get(k)=="❌ 缺失"]
    constraint_note = f"【重要约束】核心数据缺失：{', '.join(core_missing)}。置信度强制设为'低'，若清算缺失则输出'neutral'。" if core_missing else ""

    prompt = f"""你是一位拥有 15 年实战经验的加密货币首席交易员，专精于清算动力学、多空博弈定位与预期差分析。
你的任务：
- 严格遵循五步分析框架，给出逻辑自洽的策略推演与具体交易计划。
- 遇到 [N/A] 标记的数据时，该维度不得作为判断依据，且整体置信度必须为低。
- reasoning 总字数 ≤ 3000 字，结构清晰、分层明确。
- **第五步必须包含“价格路径推演：”开头的一段文字，综合运用流动性猎杀理论、行为金融学及博弈论进行推演，缺失此段将被视为分析不完整。**
- 请计算盈亏比作为参考，但不强制要求 ≥ 2:1。
- 最终输出必须为纯 JSON，且所有枚举值使用中文（做多/做空/观望等）。

【数据质量】
当前核心数据覆盖率：{coverage['coverage']:.0%}（{coverage['available']}/{coverage['total']}字段可用）。
若覆盖率低于 70%，必须将总体置信度强制设为'低'，仓位上限不超过轻仓。
当前 direction_bias 可信度：{bias_quality}。若为 untrusted，锚点约束不生效，你可独立判断。

【系统预判】
清算穿刺预判：方向 {puncture['puncture_direction']}，得分 {puncture['puncture_score']:.2f}。
诱饵风险：{puncture['lure_risk']}，期权磁吸：{puncture['pain_magnet']}。
若无数据反证，应尊重预判；若有充分反证可推翻并说明理由。
（微观数据时效检查已取消，所有高频信号（CVD、taker、OB）视为实时有效。）

{dashboard}

【市场数据】
现价：{mark_str}，ATR：{atr_str}，恐慌贪婪：{fear_greed}
LTH成本：{lth_str}，STH成本：{sth_str}，STH SOPR：{sopr_str}
稳定币趋势：{stable_str}%，OI 24h变化：{oi_chg_str}%，费率分位：{fund_pct_str}%
CVD斜率：{cvd_str}，主动买卖比(1h)：{taker_str}
24h期货净流：{nf24h_str}M，现货24h净流：{spot24_str}M，背离度：{spot_div_str}
上方清算：{abv_liq_str}B，触发距{abv_trig}点
下方清算：{blw_liq_str}B，触发距{blw_trig}点，比值：{liq_r_str}
大单卖：{lgs_str}M，买：{lgb_str}M，压迫比：{press_str}
订单簿失衡率：{ob_imb_str}，诱饵风险：{lure_str}
期权痛点：{pain_str}，P/C比：{pc_str}，基差分位：{basis_pct_str}%
BTC.D趋势：{btc_dom_str}%，借贷利率：{borrow_str}%
交易所BTC余额变化：{exch_str} BTC
价格7日分位：{price_pct_str}%，波动因子：{vol_f_str}，CGDI分位：{cgdi_pct_str}%
{cross_context}
{constraint_note}

严格按五步分析（每步均需数据确认表 + 定性分析），方向锚点 direction_bias={direction_bias:.3f}，锚点冲突且可信时必须观望。

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
  "reasoning": "五步完整分析，必须包含价格路径推演段落，字数≤3000字。",
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

# ------------------- 审计官 -------------------
def call_reviewer(strategy: dict, data: dict, symbol: str) -> dict:
    direction_bias = data.get('direction_bias',0.0)
    coverage_info = compute_coverage(data)
    coverage_pct = coverage_info['coverage']*100
    available = coverage_info['available']
    total = coverage_info['total']
    puncture = compute_liquidation_bias(data)
    puncture_direction = puncture['puncture_direction']
    lure_risk = puncture['lure_risk']
    bias_quality = data.get('_bias_quality','reliable')
    trader_direction = strategy.get('direction','')
    trader_position_size = strategy.get('position_size','')
    rr = strategy.get('risk_reward_ratio','?')

    prompt = f"""你是一位独立的风险审计官，负责对首席交易员的策略进行无偏见的严格审计。
你的职责：
- 对照市场数据，逐项核查交易员分析中的遗漏、数据误用、逻辑断裂和反证缺失。
- 所有发现必须按“步骤/问题/数据证据/影响/严重性”格式记录。
- 最终裁决（通过/存疑/驳回）必须仅基于发现的严重性和数量，不受交易员声望影响。
- 输出必须为纯 JSON，包含完整的审计报告和严重性统计。

【审计参考数据】
- 核心数据覆盖率：{coverage_pct:.0f}%（{available}/{total}）
- 系统预判穿刺方向：{puncture_direction}，诱饵风险：{lure_risk}
- 方向锚点 direction_bias：{direction_bias}，可信度：{bias_quality}
- 交易员策略方向：{trader_direction}，仓位：{trader_position_size}，盈亏比：{rr}

【交易标的】{symbol} 【锚点】direction_bias={direction_bias:.3f}
【策略】方向：{strategy.get('direction')}，仓位：{strategy.get('position_size')}
入场：{strategy.get('entry_price_low')}-{strategy.get('entry_price_high')}
止损：{strategy.get('stop_loss')}，止盈：{strategy.get('take_profit')}
【推演】{format_reasoning(strategy.get('reasoning','无'))}

按五节模板输出审计报告：
一、遗漏指标与分析缺失
二、数据与解读错误
三、逻辑错误
四、关键反证提示
五、博弈层面审视

每条发现格式：在[步骤X]中，交易员[具体问题]。该指标显示[具体数值/信号]，若纳入分析将[强化/削弱/推翻]当前方向判断。[严重性：高/中/低]
统计严重/中等/轻度数量，并给出 max_severity。
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
            resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=4096,
                timeout=120,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content or ""
            _log_response("reviewer", prompt, content)
            rev = try_parse_reviewer_json(content)
            full_report = rev.get("full_report", str(rev))
            full_report = format_reasoning(full_report)
            if sum(rev["severity_counts"].values()) == 0 and rev.get("verdict") == "驳回":
                rev["severity_counts"]["严重"] = 1
                rev["max_severity"] = "严重"
            rev["full_report"] = full_report
            return {**rev, "_model": resp.model}
        except Exception as e:
            logger.warning(f"审计官调用失败 (尝试 {attempt+1}): {e}")
            if attempt == MAX_RETRIES-1:
                return {"verdict":"驳回","max_severity":"严重","severity_counts":{"严重":1,"中等":0,"轻度":0},"full_report":"审计官调用失败，自动驳回","_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

# ------------------- 交易委员会 -------------------
def call_judge(strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    direction_bias = data.get('direction_bias',0.0)
    prompt = f"""你是交易委员会主席，拥有最终决策权。
你的任务：
- 审议首席交易员的策略及审计官的完整报告。
- 对审计官的每一条严重指控必须逐条回应，明确采纳或驳斥的理由。
- 在三维汇聚存在分歧时，用你的市场经验作出最后平衡，但不得无视硬数据约束（如方向锚点、盈亏比底线）。
- 最终输出必须为纯 JSON，裁决字段优先使用英文值（long/short/neutral、high/medium/low）。
- 若审计严重性为“严重”，你必须推翻原策略，制定独立的最终策略并给出充分理由。
- 若维持原判或修改执行，也必须给出理由。

【标的】{symbol}，现价：{data.get('mark_price',0):.2f}，锚点：{direction_bias:.3f}

【交易员策略】
方向：{strategy.get('direction')}，仓位：{strategy.get('position_size')}，置信度：{strategy.get('confidence')}
入场：{strategy.get('entry_price_low')}-{strategy.get('entry_price_high')}
止损：{strategy.get('stop_loss')}，止盈：{strategy.get('take_profit')}
推演：{format_reasoning(strategy.get('reasoning','无'))}

【审计报告】
{format_reasoning(reviewer_report.get('full_report','无'))}
审计结论：{reviewer_report.get('verdict','未知')}，最高严重性：{reviewer_report.get('max_severity','未知')}
严重性统计：{reviewer_report.get('severity_counts',{})}

对于审计官的每一项严重指控，你必须基于实际数据回应。若驳回，需指明数据中的反证依据。若采信，需说明如何修改策略。无数据支撑的驳回视为无效。
严重指控成立必须推翻或修改策略。final_reasoning 必须逐条回应指控，并给出独立、完整的最终策略。维持原判时价格字段不能填0，必须填写实际数值。
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
  "final_reasoning": "裁决书正文，必须包含对审计指控的逐条回应以及独立制定的最终策略依据"
}}
"""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
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
            json_str = extract_json_safe(content)
            result = json.loads(json_str)
            result["final_direction"] = norm_direction(result.get("final_direction",""))
            result["final_position_size"] = norm_position_size(result.get("final_position_size",""))
            result["final_confidence"] = norm_confidence(result.get("final_confidence",""))
            sev_map = {"严重":"critical","中等":"medium","轻度":"low","无":"none"}
            raw_sev = result.get("audit_max_severity","无")
            result["audit_max_severity"] = sev_map.get(raw_sev, raw_sev)
            if result.get("final_verdict") == "维持原判":
                result["entry_price_low"] = result.get("entry_price_low") or strategy.get("entry_price_low",0)
                result["entry_price_high"] = result.get("entry_price_high") or strategy.get("entry_price_high",0)
                result["stop_loss"] = result.get("stop_loss") or strategy.get("stop_loss",0)
                result["take_profit"] = result.get("take_profit") or strategy.get("take_profit",0)
                result["execution_plan"] = result.get("execution_plan") or strategy.get("execution_plan","")
                result["risk_note"] = result.get("risk_note") or strategy.get("risk_note","")
            result["final_reasoning"] = format_reasoning(result.get("final_reasoning") or "裁决完成。")
            return {**result, "_model": resp.model}
        except Exception as e:
            logger.warning(f"委员会调用失败 (尝试 {attempt+1}): {e}")
            if attempt == MAX_RETRIES-1:
                return {"final_verdict":"推翻","final_direction":"neutral","final_confidence":"low","final_position_size":"none","entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,"execution_plan":"委员会调用失败，强制观望","risk_note":"系统故障","audit_adopted":False,"audit_max_severity":"严重","final_reasoning":"委员会调用失败，自动推翻并观望","_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

def apply_final_verdict(strategy: dict, judge_result: dict) -> dict:
    verdict = judge_result.get("final_verdict","维持原判")
    logger.info(f"应用最终决议: {verdict}")
    strategy["_judge_verdict"] = verdict
    strategy["_judge_reasoning"] = judge_result.get("final_reasoning","")
    fields = ["direction","confidence","position_size","entry_price_low","entry_price_high","stop_loss","take_profit","execution_plan","risk_note"]
    if verdict == "推翻":
        if judge_result.get("final_direction") == "neutral":
            _force_neutral(strategy, "委员会推翻并改为观望")
        else:
            for k in fields:
                if k in judge_result and judge_result[k] is not None:
                    strategy[k] = judge_result[k]
    elif verdict == "修改执行":
        for k in fields:
            if k in judge_result and judge_result[k] is not None:
                strategy[k] = judge_result[k]
    else:
        strategy["risk_note"] = judge_result.get("risk_note") or strategy.get("risk_note","")
        strategy["execution_plan"] = judge_result.get("execution_plan") or strategy.get("execution_plan","")
    return strategy