"""
deepseek.py — 三角色闭环 (最终版)
- 提示词极简，无重复要求
- 清算穿刺路径修正 (基于 CVD + taker 推断近期轨迹)
- 审计官文本解析确保统计正确
- 委员会独立裁决、逐条回应
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

# ---------- 标准化 ----------
def norm_dir(raw): return raw if raw in VALID_DIRECTIONS else {"做多":"long","做空":"short","观望":"neutral"}.get(raw.strip().lower(),"neutral")
def norm_conf(raw): return raw if raw in VALID_CONFIDENCES else {"高":"high","中":"medium","低":"low"}.get(raw.strip().lower(),"medium")
def norm_pos(raw):  return raw if raw in VALID_POSITION_SIZES else {"重仓":"heavy","中仓":"medium","轻仓":"light","无":"none"}.get(raw.strip().lower(),"none")

def _force_neutral(s, reason):
    s.update({"direction":"neutral","confidence":"low","position_size":"none",
              "entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,
              "execution_plan":"","risk_note":f"观望。{reason}",
              "reasoning":(s.get("reasoning","")+f"\n[系统强制观望，原因：{reason}]").strip()})

def _log_response(role, prompt, content):
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/{role}_{ts}.json","w",encoding="utf-8") as f:
            json.dump({"prompt":prompt,"content":content},f,ensure_ascii=False,indent=2)
    except: pass

def extract_json_safe(content):
    m = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if m: return m.group(1).strip()
    m = re.search(r'```\s*([\s\S]*?)\s*```', content)
    if m: return m.group(1).strip()
    start = content.find('{')
    if start==-1: raise ValueError("未找到 JSON")
    count = 0
    for i,c in enumerate(content[start:],start):
        if c=='{': count+=1
        elif c=='}':
            count-=1
            if count==0: return content[start:i+1].strip()
    raise ValueError("JSON 未闭合")

def validate_strategy(s, data=None):
    d = s.get("direction")
    if d not in VALID_DIRECTIONS: return False,f"无效方向:{d}"
    if data:
        atr=data.get("atr_15m",0); mark=data.get("mark_price",0)
        if (data.get("above_liq",0)<=0 and data.get("below_liq",0)<=0) and d!="neutral":
            _force_neutral(s,"清算缺失"); return True,""
        if atr<=0 or mark<=0:
            if d!="neutral": _force_neutral(s,"缺失ATR/价格"); return True,""
        bias_q=data.get("_bias_quality","reliable"); bias=data.get("direction_bias",0.0)
        if bias_q in ("reliable","degraded") and abs(bias)>0.4 and d!="neutral":
            if (bias>0 and d=="short") or (bias<0 and d=="long"):
                _force_neutral(s,f"锚点冲突({bias:.3f})"); return True,""
    if d=="neutral":
        for f in ["entry_price_low","entry_price_high","stop_loss","take_profit"]: s[f]=0
        s["position_size"]="none"
        if not s.get("execution_plan"): s["execution_plan"]="等待触发条件"
        return True,""
    for f in ["entry_price_low","entry_price_high","stop_loss","take_profit"]:
        if not isinstance(s.get(f),(int,float)) or float(s[f])<=0: return False,f"缺少 {f}"
    return True,""

# ---------- 时效注入 ----------
def _inject_ages(data):
    now = time.time()
    for ts_key, age_key in [("ob_imbalance_ts","ob_age"),("taker_ratio_ts","taker_age"),
                            ("cvd_slope_ts","cvd_age"),("large_order_ts","large_order_age"),
                            ("liquidation_ts","liq_age")]:
        ts = data.get(ts_key)
        data[age_key] = (now-ts) if (ts and ts>0) else float('inf')

# ---------- 清算穿刺 (含路径修正) ----------
def compute_liquidation_bias(data):
    liq_r = data.get('liq_ratio',1.0)
    cvd = data.get('cvd_slope',0.0)
    taker = data.get('taker_ratio_1h',0.5)
    ob_imb = data.get('orderbook_imbalance',0.0)
    ob_age = data.get('ob_age',float('inf'))
    press = data.get('large_order_pressure',0.0)
    pain = data.get('max_pain',0.0)
    atr = data.get('atr',0.0)
    mark = data.get('mark_price',0.0)

    if ob_age>30: ob_imb=0.0
    score = (liq_r-1)*0.4 + (1 if cvd>0 else -1)*0.3 + (taker-0.5)*0.3

    # -------- 价格路径修正 (新增) --------
    # 推断近期轨迹：CVD>0且taker>1 → 向上穿越；CVD<0且taker<1 → 向下穿越；否则无法判断
    trajectory_modifier = 1.0
    trajectory_note = "无路径修正"
    if cvd > 0 and taker > 1:
        trajectory_note = "推断近期向上穿越"
    elif cvd < 0 and taker < 1:
        trajectory_note = "推断近期向下穿越"

    if trajectory_note == "推断近期向下穿越" and score > 0.15:
        # 预判向上穿刺，但价格刚跌下来 → 上方清算可能已被触发或阻力极大
        trajectory_modifier = 0.5
        trajectory_note += "，向上穿刺可信度减半(上方清算可能已被触发或存在解套抛压)"
    elif trajectory_note == "推断近期向上穿越" and score < -0.15:
        trajectory_modifier = 0.5
        trajectory_note += "，向下穿刺可信度减半(下方清算可能已被触发或存在空头解套)"
    elif trajectory_note in ["推断近期向上穿越","推断近期向下穿越"]:
        trajectory_note += "，穿刺方向与近期轨迹一致，可信度不变"

    score *= trajectory_modifier
    # ----------------------------------------

    direction = 'balanced'
    if score>0.15: direction='up'
    elif score<-0.15: direction='down'
    lure = (direction=='up' and press<-0.5) or (direction=='down' and press>0.5)
    pain_eff = False
    if atr>0 and pain>0 and abs(pain-mark)<1.0*atr:
        if (direction=='up' and pain>mark) or (direction=='down' and pain<mark): pain_eff=True
    return {
        'puncture_direction':direction,
        'puncture_score':score,
        'lure_risk':lure,
        'pain_magnet':pain_eff,
        'trajectory_note':trajectory_note
    }

# ---------- 微观质量 + 仪表盘 + 覆盖率 (保持简洁) ----------
def assess_micro_quality(data):
    checks = {"orderbook_fresh":data.get("ob_age",float('inf'))<30,"taker_fresh":data.get("taker_age",float('inf'))<60,
              "cvd_fresh":data.get("cvd_age",float('inf'))<300,"large_order_fresh":data.get("large_order_age",float('inf'))<300,
              "liquidation_fresh":data.get("liq_age",float('inf'))<600}
    fresh = sum(checks.values())
    return {**checks,"overall":"good" if fresh>=4 else ("degraded" if fresh>=2 else "poor")}

def build_expectation_dashboard(data):
    basis_ann=data.get('basis_annualized',0); basis_med=data.get('basis_median',8); fund_pct=data.get('funding_percentile',50)
    cgdi_pct=data.get('cgdi_percentile',50); st_flow=data.get('stablecoin_trend_7d',0); btc_dom=data.get('btc_dominance_trend_7d',0)
    borrow=data.get('borrow_rate',0)*100; pc=data.get('put_call_ratio',1.0); price_pct=data.get('price_percentile',50); vol_f=data.get('vol_factor',1.0)
    return f"""【预期定价仪表盘】
| 指标 | 当前值 | 历史基线 | 定价了什么？ |
|------|--------|----------|------------|
| 3月基差年化 | {basis_ann:.1f}% | {basis_med:.1f}% | 期货溢价程度 |
| 资金费率分位 | {fund_pct:.0f}% | 50% | 多头支付意愿 |
| CGDI分位 | {cgdi_pct:.0f}% | 50% | 综合贪婪度 |
| 稳定币净流7d | {st_flow:+.1f}% | +0.5% | 资金面松紧 |
| BTC.D趋势7d | {btc_dom:+.1f}% | 0% | 风险偏好 |
| 借贷利率 | {borrow:.2f}% | 均值 | 杠杆紧张度 |
| P/C比 | {pc:.3f} | 0.7 | >1恐慌对冲 |
| 价格7日分位 | {price_pct:.0f}% | 50% | 超买/超卖 |
| 波动因子 | {vol_f:.2f} | 1.0 | 不确定性定价 |
预期差分析必须回答：1.极端方向(贪婪/恐惧)依据哪些指标？2.找出两个矛盾指标构成预期差。3.价格朝矛盾方向移动1ATR谁最意外？4.结论必须基于矛盾证据并与清算预判、多空博弈交叉核对。"""

CORE_KEYS = ['mark_price','atr','above_liq','below_liq','liq_ratio','cvd_slope','taker_ratio_1h','oi_change_24h',
             'funding_percentile','orderbook_imbalance','large_order_pressure','max_pain','put_call_ratio',
             'basis_percentile','stablecoin_trend_7d','cgdi_percentile','fear_greed','lth_realized_price','sth_realized_price','sth_sopr']

def compute_coverage(data):
    total = len(CORE_KEYS)
    available = sum(1 for k in CORE_KEYS if data.get(k) is not None)
    return {"available":available,"total":total,"coverage":available/total if total>0 else 0.0}

# ------------------- 主提示词 -------------------
def build_prompt(data, symbol, eth_data=None, cross_symbol=None):
    if cross_symbol is None: cross_symbol = "ETH" if symbol=="BTC" else "BTC"
    _inject_ages(data)
    coverage = compute_coverage(data)

    def sv(key,default=0.0,scale=1.0,fmt=".2f"):
        raw = data.get(key)
        if raw is None: return ("[N/A]",True)
        try: val=float(raw)*scale
        except: return ("[N/A]",True)
        if math.isnan(val) or math.isinf(val): return ("[N/A]",True)
        try: return (f"{val:{fmt}}",False)
        except: return ("[N/A]",True)

    # 提取字段 (全部保留，篇幅问题省略部分重复声明)
    mark_str,_ = sv('mark_price',fmt=".2f"); atr_str,_ = sv('atr',fmt=".2f")
    fear_greed = data.get('fear_greed',50)
    lth_str,_ = sv('lth_realized_price',fmt=".2f"); sth_str,_ = sv('sth_realized_price',fmt=".2f")
    sopr_str,_ = sv('sth_sopr',1.0,fmt=".3f"); stable_str,_ = sv('stablecoin_trend_7d',fmt="+.1f")
    oi_chg_str,_ = sv('oi_change_24h',fmt="+.1f"); fund_pct_str,_ = sv('funding_percentile',50,fmt=".0f")
    cvd_str,_ = sv('cvd_slope',fmt=".4f"); taker_str,_ = sv('taker_ratio_1h',fmt=".3f")
    nf24h_str,_ = sv('netflow_24h',scale=1/1e6,fmt=".1f")
    abv_liq_str,_ = sv('above_liq',scale=1/1e9,fmt=".2f"); blw_liq_str,_ = sv('below_liq',scale=1/1e9,fmt=".2f")
    liq_r_str,_ = sv('liq_ratio',fmt=".2f")
    abv_trig = data.get('above_trigger','N/A'); blw_trig = data.get('below_trigger','N/A')
    lgs_str,_ = sv('large_sell_value',scale=1/1e6,fmt=".1f"); lgb_str,_ = sv('large_buy_value',scale=1/1e6,fmt=".1f")
    press_str,_ = sv('large_order_pressure',fmt=".3f"); ob_imb_str,_ = sv('orderbook_imbalance',fmt=".3f")
    lure_str,_ = sv('lure_risk_factor',fmt=".2f"); pain_str,_ = sv('max_pain',fmt=".2f")
    pc_str,_ = sv('put_call_ratio',fmt=".4f"); basis_pct_str,_ = sv('basis_percentile',50,fmt=".0f")
    btc_dom_str,_ = sv('btc_dominance_trend_7d',fmt="+.1f"); borrow_str,_ = sv('borrow_rate',scale=100,fmt=".2f")
    exch_str,_ = sv('exchange_btc_change_24h',fmt="+.0f"); spot24_str,_ = sv('spot_netflow_24h',scale=1/1e6,fmt=".1f")
    spot_div_str,_ = sv('spot_vs_futures_divergence',fmt=".2f"); top_ls_str,_ = sv('top_ls_percentile',50,fmt=".0f")
    price_pct_str,_ = sv('price_percentile',50,fmt=".0f"); vol_f_str,_ = sv('vol_factor',1.0,fmt=".2f")
    cgdi_pct_str,_ = sv('cgdi_percentile',50,fmt=".0f"); direction_bias = data.get('direction_bias',0.0)
    bias_quality = data.get('_bias_quality','reliable')

    puncture = compute_liquidation_bias(data)
    micro_q = assess_micro_quality(data)
    dashboard = build_expectation_dashboard(data)

    cross_context = ""
    if eth_data:
        cross_context = f"""【跨币种数据（{cross_symbol}）—— 用于第二步与第四步】
| 指标 | {cross_symbol} 当前值 | {symbol} 当前值 |
|------|---------------------|-----------------|
| 清算比值 | {eth_data.get('liq_ratio',0):.2f} | {data.get('liq_ratio',1):.2f} |
| CVD斜率 | {eth_data.get('cvd_slope',0):.4f} | {data.get('cvd_slope',0):.4f} |
| OI 24h变化 | {eth_data.get('oi_change_24h',0):+.1f}% | {data.get('oi_change_24h',0):+.1f}% |
| 顶多空分位 | {eth_data.get('top_ls_percentile',50):.0f}% | {data.get('top_ls_percentile',50):.0f}% |
规则：第二步多空论据可引用，交叉质询矛盾信号须作为攻击依据。第四步信号定性：清算方向一致→系统性趋势，仓位不变；矛盾→仓位降一级。"""
    else:
        cross_context = "\n【跨币种数据不可用】仓位上限自动下调一级，置信度上限为'中'。"

    core_missing = [k for k in ["kline","heatmap","cvd"] if data.get("data_quality",{}).get(k)=="❌ 缺失"]
    constraint_note = f"核心数据缺失：{', '.join(core_missing)}。置信度强制'低'，若清算缺失输出'neutral'。" if core_missing else ""

    prompt = f"""你是一位拥有 15 年实战经验的加密货币首席交易员。遇到 [N/A] 标记的数据时，该维度不作为依据，且置信度必须为低。reasoning 总字数 ≤ 3000 字。

【数据质量】核心数据覆盖率：{coverage['coverage']:.0%}({coverage['available']}/{coverage['total']})。低于70%时强制置信度'低'，仓位上限轻仓。direction_bias 可信度：{bias_quality}，untrusted时不生效。

【系统预判】清算穿刺方向：{puncture['puncture_direction']}，得分 {puncture['puncture_score']:.2f}。诱饵风险：{puncture['lure_risk']}，期权磁吸：{puncture['pain_magnet']}。价格路径修正：{puncture['trajectory_note']}。若无数据反证应尊重预判，有充分反证可推翻并说明理由。微观新鲜度：{micro_q['overall']}，poor时高频信号权重降为0。

{dashboard}

【市场数据】
现价：{mark_str}，ATR：{atr_str}，恐慌贪婪：{fear_greed}
LTH成本：{lth_str}，STH成本：{sth_str}，STH SOPR：{sopr_str}
稳定币趋势：{stable_str}%，OI 24h变化：{oi_chg_str}%，费率分位：{fund_pct_str}%
CVD斜率：{cvd_str}，主动买卖比(1h)：{taker_str}
24h期货净流：{nf24h_str}M，现货24h净流：{spot24_str}M，背离度：{spot_div_str}
上方清算：{abv_liq_str}B，触发距{abv_trig}点；下方清算：{blw_liq_str}B，触发距{blw_trig}点，比值：{liq_r_str}
大单卖：{lgs_str}M，买：{lgb_str}M，压迫比：{press_str}
订单簿失衡率：{ob_imb_str}，诱饵风险：{lure_str}
期权痛点：{pain_str}，P/C比：{pc_str}，基差分位：{basis_pct_str}%
BTC.D趋势：{btc_dom_str}%，借贷利率：{borrow_str}%，交易所BTC余额变化：{exch_str} BTC
价格7日分位：{price_pct_str}%，波动因子：{vol_f_str}，CGDI分位：{cgdi_pct_str}%
{cross_context}
{constraint_note}

严格按五步分析(每步数据确认表+定性分析)，锚点 direction_bias={direction_bias:.3f}，盈亏比≥2:1。第五步必须输出价格路径推演。
输出纯JSON，枚举值用中文。

【输出JSON】
{{
  "direction": "做多/做空/观望",
  "confidence": "高/中/低",
  "position_size": "重仓/中仓/轻仓/无",
  "entry_price_low": 0.0, "entry_price_high": 0.0,
  "stop_loss": 0.0, "take_profit": 0.0,
  "execution_plan": "一句话指令",
  "reasoning": "完整五步推演，含价格路径推演段落。",
  "risk_note": "", "risk_reward_ratio": 0.0,
  "vote_result": {{"清算":"","博弈":"","预期差":"","一致组数":0,"方向":""}}
}}
"""
    return prompt

# ------------------- 交易员 -------------------
def call_trader(prompt):
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),base_url="https://api.deepseek.com",timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=FAST_MODEL,messages=[{"role":"user","content":prompt}],max_tokens=16384,timeout=TIMEOUT_SECONDS)
            content = resp.choices[0].message.content or ""
            _log_response("trader",prompt,content)
            if not content.strip(): raise ValueError("空响应")
            s = json.loads(extract_json_safe(content))
            s["direction"]=norm_dir(s.get("direction","")); s["position_size"]=norm_pos(s.get("position_size",""))
            s["confidence"]=norm_conf(s.get("confidence",""))
            s.setdefault("reasoning",""); s.setdefault("risk_note",""); s.setdefault("execution_plan","")
            s["_model_used"]=resp.model
            return s
        except Exception as e:
            logger.warning(f"交易员失败: {e}")
            if attempt==MAX_RETRIES-1:
                return {"direction":"neutral","confidence":"low","position_size":"none","entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,"execution_plan":"调用失败","reasoning":"调用失败","risk_note":"","_model_used":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

# ------------------- 审计官 -------------------
def call_reviewer(strategy, data, symbol):
    direction_bias = data.get('direction_bias',0.0)
    micro_q = assess_micro_quality(data)
    ages_info = f"OB{data.get('ob_age','?')}s Taker{data.get('taker_age','?')}s CVD{data.get('cvd_age','?')}s Lg{data.get('large_order_age','?')}s Liq{data.get('liq_age','?')}s"
    prompt = f"""你是独立风险审计官。逐项核查交易员策略，每条发现严格按“[严重性：高/中/低]”标记结尾。输出纯JSON。

标的：{symbol} 锚点：{direction_bias:.3f} 微观时效：{micro_q['overall']}({ages_info})
策略：方向{strategy.get('direction')} 仓位{strategy.get('position_size')} 入场{strategy.get('entry_price_low')}-{strategy.get('entry_price_high')} 止损{strategy.get('stop_loss')} 止盈{strategy.get('take_profit')}
推演：{strategy.get('reasoning','无')}

按五节输出审计报告，包含完整 severity_counts。
【输出JSON】{{"verdict":"通过/存疑/驳回","max_severity":"严重/中等/轻度/无","severity_counts":{{"严重":0,"中等":0,"轻度":0}},"full_report":"完整审计文本"}}
"""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),base_url="https://api.deepseek.com",timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=FAST_MODEL,messages=[{"role":"user","content":prompt}],max_tokens=4096,timeout=120)
            content = resp.choices[0].message.content or ""
            rev = json.loads(extract_json_safe(content))
            full = rev.get("full_report",str(rev))
            cnt = {"严重":0,"中等":0,"轻度":0}
            for line in full.split('\n'):
                if '严重性：高' in line: cnt["严重"]+=1
                elif '严重性：中' in line: cnt["中等"]+=1
                elif '严重性：低' in line: cnt["轻度"]+=1
            rev["severity_counts"] = cnt
            rev["max_severity"] = "严重" if cnt["严重"]>0 else ("中等" if cnt["中等"]>0 else ("轻度" if cnt["轻度"]>0 else "无"))
            rev["verdict"] = "驳回" if cnt["严重"]>0 else ("存疑" if cnt["中等"]+cnt["轻度"]>0 else rev.get("verdict","通过"))
            rev["full_report"] = full
            return {**rev,"_model":resp.model}
        except Exception as e:
            logger.warning(f"审计官失败: {e}")
            if attempt==MAX_RETRIES-1: return {"verdict":"驳回","max_severity":"严重","severity_counts":{"严重":1},"full_report":"审计官调用失败","_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

# ------------------- 委员会 -------------------
def call_judge(strategy, reviewer_report, data, symbol):
    direction_bias = data.get('direction_bias',0.0)
    prompt = f"""你是交易委员会主席。审议策略及审计报告，逐条回应严重指控，明确采纳或驳斥理由。维持原判时价格字段不能填0。输出纯JSON，裁决字段用英文值。

标的：{symbol} 现价：{data.get('mark_price',0):.2f} 锚点：{direction_bias:.3f}
策略：方向{strategy.get('direction')} 仓位{strategy.get('position_size')} 入场{strategy.get('entry_price_low')}-{strategy.get('entry_price_high')} 止损{strategy.get('stop_loss')} 止盈{strategy.get('take_profit')}
审计：{reviewer_report.get('full_report','无')} 结论{reviewer_report.get('verdict')} 严重性{reviewer_report.get('max_severity')}

【输出JSON】{{"final_verdict":"维持原判/修改执行/推翻","final_direction":"long/short/neutral","final_confidence":"high/medium/low","final_position_size":"heavy/medium/light/none","entry_price_low":0.0,"entry_price_high":0.0,"stop_loss":0.0,"take_profit":0.0,"execution_plan":"","risk_note":"","audit_adopted":true,"audit_max_severity":"严重/中等/轻度/无","final_reasoning":"裁决书正文"}}
"""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),base_url="https://api.deepseek.com",timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=REASONING_MODEL,messages=[{"role":"user","content":prompt}],max_tokens=16384,timeout=120)
            content = resp.choices[0].message.content or ""
            result = json.loads(extract_json_safe(content))
            result["final_direction"]=norm_dir(result.get("final_direction",""))
            result["final_position_size"]=norm_pos(result.get("final_position_size",""))
            result["final_confidence"]=norm_conf(result.get("final_confidence",""))
            sev_map={"严重":"critical","中等":"medium","轻度":"low","无":"none"}
            result["audit_max_severity"]=sev_map.get(result.get("audit_max_severity","无"),"none")
            if result.get("final_verdict")=="维持原判":
                result["entry_price_low"]=result.get("entry_price_low") or strategy.get("entry_price_low",0)
                result["entry_price_high"]=result.get("entry_price_high") or strategy.get("entry_price_high",0)
                result["stop_loss"]=result.get("stop_loss") or strategy.get("stop_loss",0)
                result["take_profit"]=result.get("take_profit") or strategy.get("take_profit",0)
                result["execution_plan"]=result.get("execution_plan") or strategy.get("execution_plan","")
                result["risk_note"]=result.get("risk_note") or strategy.get("risk_note","")
            result["final_reasoning"]=result.get("final_reasoning") or "裁决完成。"
            return {**result,"_model":resp.model}
        except Exception as e:
            logger.warning(f"委员会失败: {e}")
            if attempt==MAX_RETRIES-1:
                return {"final_verdict":"推翻","final_direction":"neutral","final_confidence":"low","final_position_size":"none","entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,"execution_plan":"委员会调用失败","risk_note":"系统故障","audit_adopted":False,"audit_max_severity":"critical","final_reasoning":"委员会调用失败","_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

def apply_final_verdict(strategy, judge_result):
    verdict = judge_result.get("final_verdict","维持原判")
    logger.info(f"最终决议: {verdict}")
    strategy["_judge_verdict"] = verdict
    strategy["_judge_reasoning"] = judge_result.get("final_reasoning","")
    fields = ["direction","confidence","position_size","entry_price_low","entry_price_high","stop_loss","take_profit","execution_plan","risk_note"]
    if verdict == "推翻":
        if judge_result.get("final_direction")=="neutral":
            _force_neutral(strategy,"委员会推翻")
        else:
            for k in fields:
                if k in judge_result and judge_result[k] is not None: strategy[k]=judge_result[k]
    elif verdict == "修改执行":
        for k in fields:
            if k in judge_result and judge_result[k] is not None: strategy[k]=judge_result[k]
    else:
        strategy["risk_note"] = judge_result.get("risk_note") or strategy.get("risk_note","")
        strategy["execution_plan"] = judge_result.get("execution_plan") or strategy.get("execution_plan","")
    return strategy