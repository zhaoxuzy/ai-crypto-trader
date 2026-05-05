"""
deepseek.py — 生产级三角色闭环 (审计提示词恢复版)
- 保留定价仪表盘、穿刺预判、覆盖率、时效约束
- 审计官提示词恢复简洁定义，不添加硬规则清单
- 修复 overall 变量未定义异常
- 修复盈亏比硬约束（已移除）
- 修复 reasoning 格式化和换行问题
- 修复审计严重性统计
- 增强 JSON 提取，审计官输出纯 JSON 失败时自动 fallback
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
    if not text:
        return text
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

def _log_audit_raw(content: str):
    """将审计官原始响应写入日志以便排查"""
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/audit_raw_{ts}.txt", "w", encoding="utf-8") as f:
            f.write(content)
    except:
        pass

def extract_json_safe(content: str) -> str:
    # 优先匹配 ```json ... ```
    m = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if m:
        return m.group(1).strip()
    # 匹配 ``` ... ``` 无语言标识
    m = re.search(r'```\s*([\s\S]*?)\s*```', content)
    if m:
        return m.group(1).strip()
    # 从第一个 '{' 到最后一个 '}' 提取，尝试修复常见错误
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = content[start:end+1].strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            # 简单修复：移除尾部的逗号
            fixed = re.sub(r',\s*}', '}', candidate)
            fixed = re.sub(r',\s*]', ']', fixed)
            try:
                json.loads(fixed)
                return fixed
            except:
                pass
    raise ValueError("未找到有效 JSON")

def _force_neutral(s: dict, reason: str):
    s.update({"direction":"neutral","confidence":"低","position_size":"none",
              "entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,
              "execution_plan":"","reasoning":(s.get("reasoning","")+f"\n\n[系统强制观望，原因：{reason}]").strip(),
              "risk_note":f"观望。{reason}"})

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

# ---------- 时效注入 ----------
def _inject_ages(data: dict):
    now = time.time()
    for ts_key, age_key in [("ob_imbalance_ts","ob_age"),("taker_ratio_ts","taker_age"),
                            ("cvd_slope_ts","cvd_age"),("large_order_ts","large_order_age"),
                            ("liquidation_ts","liq_age")]:
        ts = data.get(ts_key)
        data[age_key] = (now - ts) if (ts and ts>0) else float('inf')

# ---------- 清算穿刺 ----------
def compute_liquidation_bias(data: dict) -> dict:
    liq_r = data.get('liq_ratio',1.0)
    cvd = data.get('cvd_slope',0.0)
    taker = data.get('taker_ratio_1h',0.5)
    ob_imb = data.get('orderbook_imbalance',0.0)
    ob_age = data.get('ob_age', float('inf'))
    press = data.get('large_order_pressure',0.0)
    pain = data.get('max_pain',0.0)
    atr = data.get('atr',0.0)
    mark = data.get('mark_price',0.0)

    if ob_age>30: ob_imb=0.0
    score = (liq_r-1.0)*0.4 + (1 if cvd>0 else -1)*0.3 + (taker-0.5)*0.3
    direction = 'balanced'
    if score>0.15: direction='up'
    elif score<-0.15: direction='down'
    lure = (direction=='up' and press<-0.5) or (direction=='down' and press>0.5)
    pain_eff = False
    if atr>0 and pain>0 and abs(pain-mark)<1.0*atr:
        if (direction=='up' and pain>mark) or (direction=='down' and pain<mark): pain_eff=True
    return {'puncture_direction':direction,'puncture_score':score,'lure_risk':lure,'pain_magnet':pain_eff}

# ---------- 微观质量 ----------
def assess_micro_quality(data: dict) -> dict:
    checks = {
        "orderbook_fresh": data.get("ob_age",float('inf'))<30,
        "taker_fresh": data.get("taker_age",float('inf'))<60,
        "cvd_fresh": data.get("cvd_age",float('inf'))<300,
        "large_order_fresh": data.get("large_order_age",float('inf'))<300,
        "liquidation_fresh": data.get("liq_age",float('inf'))<600,
    }
    fresh_count = sum(checks.values())
    overall = "good" if fresh_count>=4 else ("degraded" if fresh_count>=2 else "poor")
    return {**checks, "overall":overall}

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

# ------------------- 交易员提示词 -------------------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    # ...（此处省略与之前完全相同的 build_prompt 代码，未做任何修改）...
    # 因篇幅原因在此省略，但必须保留原始完整实现。
    pass

# ------------------- 首席交易员 -------------------
def call_trader(prompt: str) -> dict:
    # ...（与原版完全相同，无修改）...
    pass

# ------------------- 审计官 (强化纯 JSON + fallback) -------------------
def call_reviewer(strategy: dict, data: dict, symbol: str) -> dict:
    direction_bias = data.get('direction_bias',0.0)
    micro_q = assess_micro_quality(data)
    overall = micro_q['overall']
    ob_age = data.get('ob_age','?')
    taker_age = data.get('taker_age','?')
    cvd_age = data.get('cvd_age','?')
    large_age = data.get('large_order_age','?')
    liq_age = data.get('liq_age','?')
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
    ages_info = f"订单簿{ob_age}s，主动成交{taker_age}s，CVD{cvd_age}s，大单{large_age}s，清算{liq_age}s"

    prompt = f"""你是一位独立的风险审计官，负责对首席交易员的策略进行无偏见的严格审计。
你的职责：
- 对照市场数据，逐项核查交易员分析中的遗漏、数据误用、逻辑断裂和反证缺失。
- 所有发现必须按“步骤/问题/数据证据/影响/严重性”格式记录。
- 最终裁决（通过/存疑/驳回）必须仅基于发现的严重性和数量，不受交易员声望影响。
- **输出格式：纯 JSON，不含任何代码块标记（如 ```json），不含任何前置或后置说明。**
- 输出必须以 `{{` 开头，以 `}}` 结尾。

【审计参考数据】
- 市场数据新鲜度：{overall}（{ages_info}）
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

【输出JSON范例】
{{
  "verdict": "驳回",
  "max_severity": "严重",
  "severity_counts": {{"严重":2,"中等":1,"轻度":0}},
  "full_report": "一、遗漏指标...\\n二、数据与解读错误...\\n..."
}}
现在，只输出纯JSON：
"""

    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=4096,
                timeout=120
            )
            content = resp.choices[0].message.content or ""

            # 可选调试日志
            if os.getenv("DEBUG_AUDIT", "").lower() == "true":
                _log_audit_raw(content)

            # ---------- JSON 提取与 fallback ----------
            try:
                json_str = extract_json_safe(content)
                rev = json.loads(json_str)
            except Exception as parse_err:
                logger.warning(f"审计官 JSON 提取失败: {parse_err}")
                # 将原始内容作为报告，扫描严重性
                full_fallback = content.strip()
                if not full_fallback:
                    raise ValueError("审计官返回空内容")
                cnt = {"严重": 0, "中等": 0, "轻度": 0}
                for line in full_fallback.split('\n'):
                    if '严重性：高' in line or '严重性：严重' in line:
                        cnt["严重"] += 1
                    elif '严重性：中' in line or '严重性：中等' in line:
                        cnt["中等"] += 1
                    elif '严重性：低' in line or '严重性：轻度' in line:
                        cnt["轻度"] += 1
                if cnt["严重"] > 0:
                    verdict = "驳回"
                    max_sev = "严重"
                elif cnt["中等"] > 0:
                    verdict = "存疑"
                    max_sev = "中等"
                elif cnt["轻度"] > 0:
                    verdict = "存疑"
                    max_sev = "轻度"
                else:
                    verdict = "存疑"
                    max_sev = "轻度"
                rev = {
                    "verdict": verdict,
                    "max_severity": max_sev,
                    "severity_counts": cnt,
                    "full_report": full_fallback
                }
                logger.info("审计官 JSON 提取失败，已根据文本生成 fallback 报告")
                return {**rev, "_model": resp.model}

            # ---------- 正常 JSON 解析成功后的处理 ----------
            full_report = rev.get("full_report", str(rev))
            full_report = format_reasoning(full_report)

            severity_counts = rev.get("severity_counts", {})
            if not isinstance(severity_counts, dict) or not severity_counts:
                cnt = {"严重": 0, "中等": 0, "轻度": 0}
                for line in full_report.split('\n'):
                    if '严重性：高' in line or '严重性：严重' in line:
                        cnt["严重"] += 1
                    elif '严重性：中' in line or '严重性：中等' in line:
                        cnt["中等"] += 1
                    elif '严重性：低' in line or '严重性：轻度' in line:
                        cnt["轻度"] += 1
                rev["severity_counts"] = cnt
            else:
                rev["severity_counts"] = {
                    "严重": severity_counts.get("严重", 0) or severity_counts.get("critical", 0),
                    "中等": severity_counts.get("中等", 0) or severity_counts.get("medium", 0),
                    "轻度": severity_counts.get("轻度", 0) or severity_counts.get("low", 0),
                }

            cnt = rev["severity_counts"]
            if cnt["严重"] > 0:
                rev["max_severity"] = "严重"
                rev["verdict"] = "驳回"
            elif cnt["中等"] > 0:
                rev["max_severity"] = "中等"
                if rev.get("verdict") not in ("驳回","通过"):
                    rev["verdict"] = "存疑"
            elif cnt["轻度"] > 0:
                rev["max_severity"] = "轻度"
                if rev.get("verdict") not in ("驳回","通过"):
                    rev["verdict"] = "存疑"
            else:
                rev["max_severity"] = "无"
                rev["verdict"] = rev.get("verdict", "通过")

            rev["full_report"] = full_report
            return {**rev, "_model": resp.model}

        except Exception as e:
            logger.warning(f"审计官调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {
                    "verdict": "驳回",
                    "max_severity": "严重",
                    "severity_counts": {"严重": 1, "中等": 0, "轻度": 0},
                    "full_report": f"审计官三次调用均失败，自动驳回。错误: {e}",
                    "_model": "fallback"
                }
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

# ------------------- 交易委员会 -------------------
def call_judge(strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    # ...（与原版完全相同，无修改）...
    pass

def apply_final_verdict(strategy: dict, judge_result: dict) -> dict:
    # ...（与原版完全相同，无修改）...
    pass