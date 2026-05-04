"""
deepseek.py — 三段式分析闭环（交易员 · 审计官 · 委员会）
- 交易员：周期定位 + 三组投票 + 战术定位 + 逻辑终审
- 审计官：规则核查 + 严重等级标签（[严重]/[中等]/[轻度]/通过）
- 委员会：权衡裁决，输出最终可执行策略
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
def _log_response(prompt, content, reasoning=None):
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/deepseek_{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "content": content, "reasoning": reasoning}, f, ensure_ascii=False, indent=2)
    except:
        pass

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
    logger.warning("JSON 严重损坏，暴力修补")
    return content[start:] + '}}'

def _force_neutral(s, reason):
    s.update({"direction": "neutral", "confidence": "低", "position_size": "无",
              "entry_price_low": 0, "entry_price_high": 0, "stop_loss": 0, "take_profit": 0,
              "execution_plan": "", "risk_note": f"观望。{reason}"})
    s["reasoning"] = (s.get("reasoning", "") + f"\n\n[系统强制观望：{reason}]").strip()

def validate_strategy(s, data=None):
    if s.get("direction") == "neutral":
        for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
            s[f] = 0
        s["position_size"] = "无"
        if not s.get("execution_plan"): s["execution_plan"] = "观望"
        return True, ""
    return True, ""

# ------------------- 交易员提示词 -------------------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    if cross_symbol is None:
        cross_symbol = "ETH" if symbol == "BTC" else "BTC"

    def sv(key, default=0.0, scale=1.0, fmt=".2f"):
        raw = data.get(key)
        if raw is None: return ("缺失", True)
        try: return (f"{float(raw)*scale:{fmt}}", False)
        except: return ("缺失", True)

    # 提取所有字段（省略重复代码，实际包含完整字段）
    mark_price_str, _ = sv('mark_price', fmt=".2f")
    atr_str, _ = sv('atr', fmt=".2f")
    price_percentile_str, _ = sv('price_percentile', 50.0, fmt=".0f")
    cgdi_str, _ = sv('cgdi_current', fmt=".0f")
    cgdi_perc_str, _ = sv('cgdi_percentile', 50.0, fmt=".0f")
    fear_greed = data.get('fear_greed', 50)
    above_trigger = data.get('above_trigger', 'N/A')
    below_trigger = data.get('below_trigger', 'N/A')
    lth_rp_str, _ = sv('lth_realized_price', fmt=".2f")
    sth_rp_str, _ = sv('sth_realized_price', fmt=".2f")
    sth_sopr_str, _ = sv('sth_sopr', 1.0, fmt=".3f")
    stable_trend_str, _ = sv('stablecoin_trend_7d', fmt="+.1f")
    oi_chg_str, _ = sv('oi_change_24h', fmt="+.1f")
    fund_perc_str, _ = sv('funding_percentile', 50.0, fmt=".0f")
    cvd_slope_str, _ = sv('cvd_slope', fmt=".4f")
    taker_str, _ = sv('taker_ratio_1h', fmt=".3f")
    nf24h_str, _ = sv('netflow_24h', scale=1/1e6, fmt=".1f")
    max_pain_str, _ = sv('max_pain', fmt=".2f")
    pressure_str, _ = sv('large_order_pressure', fmt=".3f")
    direction_bias = data.get('direction_bias', 0.0)

    cross_context = ""
    if eth_data:
        cross_context = f"跨币种：{cross_symbol} OI 24h变化：{eth_data.get('oi_change_24h', 0):+.1f}%，清算方向：{eth_data.get('liq_ratio', 0):.2f}"

    prompt = f"""你是加密货币首席交易员。请基于原始数据完成分析，并严格按照以下结构输出最终交易计划。

【核心规则】
- 周期约束：价值区(价格≤LTH成本)只做多；派发区(价格≥STH成本*1.3或LTH SOPR>1.2)只做空；若投票方向与周期矛盾，必须观望。
- 投票规则：三组方向必须至少两组一致，否则观望。
- 盈亏比：必须≥2:1，否则观望。
- 锚点纪律：|direction_bias|>0.4且方向相反时，强制观望；否则可降仓执行。

【{symbol} 市场数据】
现价：{mark_price_str}
ATR(4h)：{atr_str}
价格7日分位：{price_percentile_str}%
恐慌贪婪：{fear_greed}
LTH成本：{lth_rp_str}，STH成本：{sth_rp_str}，STH SOPR：{sth_sopr_str}
稳定币市值趋势：{stable_trend_str}%
OI 24h变化：{oi_chg_str}%，资金费率分位：{fund_perc_str}%
CVD斜率：{cvd_slope_str}，主动买卖比(1h)：{taker_str}
24h期货净流：{nf24h_str}M
上方清算触发：{above_trigger}点，下方：{below_trigger}点
期权最大痛点：{max_pain_str}
大单压迫比：{pressure_str}
系统锚点 direction_bias：{direction_bias:.3f}
{cross_context}

【请完成以下分析并填入最终JSON】
1. 周期定位：基于LTH/STH成本、STH SOPR、稳定币趋势、恐慌贪婪，判断当前周期阶段（价值区/拉升区/派发区/下跌区）和最大仓位上限。
2. 三组投票（必须逐项写出数据及判断）：
   - 第一组 资金流：CVD斜率、主动买卖比、24h期货净流 → 方向投票（看多/空/中性）
   - 第二组 情绪：OI 24h变化、资金费率分位、恐慌贪婪 → 方向投票
   - 第三组 链上：STH SOPR、现价vs STH成本 → 方向投票
   综合三组结果，得出最终方向（做多/空/观望）及仓位（重仓/中仓/轻仓）。
3. 战术定位：结合清算触发距、期权痛点、大单压迫比、ATR，给出入场区间、止损、止盈，必须计算盈亏比并满足≥2:1。
4. 逻辑终审：检查周期与方向是否一致、投票一致组数、盈亏比、锚点冲突，修正不合理处。

输出JSON格式：
{{
  "direction": "做多/做空/观望",
  "confidence": "高/中/低",
  "position_size": "重仓/中仓/轻仓/无",
  "entry_price_low": 0.0, "entry_price_high": 0.0,
  "stop_loss": 0.0, "take_profit": 0.0,
  "execution_plan": "一句话指令",
  "reasoning": "包含完整分析过程和投票表",
  "risk_note": "核心风险",
  "cycle_phase": "价值区/拉升区/派发区/下跌区",
  "vote_result": {{ "资金流": "看多/空/中性", "情绪": "看多/空/中性", "链上": "看多/空/中性", "一致组数": 2, "最终方向": "做多/空/观望" }},
  "risk_reward_ratio": 2.5
}}
"""
    return prompt

# ------------------- 审计官提示词 -------------------
def build_reviewer_prompt(original_strategy: dict, data: dict, symbol: str) -> str:
    direction_bias = data.get('direction_bias', 0.0)
    return f"""你是风控审计官。请核验交易员策略是否违反以下硬规则，并给出严重等级。

【硬规则清单】
1. 周期一致性：周期阶段允许的方向与最终方向是否一致？
2. 投票纪律：三组方向投票至少两组一致？不一致则必须观望。
3. 盈亏比：是否≥2:1？
4. 锚点冲突：|direction_bias|>0.4时方向必须一致，否则观望。
5. 数据引用准确性：交易员引用的数值是否正确？是否遗漏关键数据？
6. 逻辑自洽：交易员的周期判断、投票、战术是否自相矛盾？

【交易标的】{symbol}
【系统锚点】direction_bias={direction_bias:.3f}

【交易员策略】
方向：{original_strategy.get('direction')}
周期阶段：{original_strategy.get('cycle_phase', '')}
投票：{original_strategy.get('vote_result', {})}
入场：{original_strategy.get('entry_price_low')}-{original_strategy.get('entry_price_high')}
止损：{original_strategy.get('stop_loss')}，止盈：{original_strategy.get('take_profit')}
盈亏比：{original_strategy.get('risk_reward_ratio')}
推演：{original_strategy.get('reasoning', '无')}

【输出格式JSON】
{{
  "verdict": "通过/存疑/驳回",
  "max_severity": "严重/中等/轻度/无",
  "severity_counts": {{"严重":0, "中等":0, "轻度":0}},
  "full_report": "逐条核查结果，每条末尾标注[严重性：高/中/低]",
  "critical_items": ["违反硬规则条目列表"]
}}
"""

# ------------------- 委员会提示词 -------------------
def build_judge_prompt(original_strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> str:
    direction_bias = data.get('direction_bias', 0.0)
    return f"""你是交易委会主席，拥有最终决策权。交易员已进行规则数据分析，给出{初步方向}。审计官提供了补充洞察，指出[关键发现]。

请基于以下框架作出最终裁决：
1. **风险评估**：审计官揭示的信号，是新增风险还是隐藏机会？它对原决策的影响程度有多大？
2. **决策选择**：综合所有信息，你选择：
   A. 同意观望（当前信号不足或矛盾）
   B. 建议谨慎开仓（审计官发现有足够强度的新信号），并给出具体方向和仓位。交易委员会主席。请基于审计报告和交易员策略，做出最终裁决。

【交易标的】{symbol}
【系统锚点】direction_bias={direction_bias:.3f}

【交易员策略】
方向：{original_strategy.get('direction')}，仓位：{original_strategy.get('position_size')}
入场：{original_strategy.get('entry_price_low')}-{original_strategy.get('entry_price_high')}
止损：{original_strategy.get('stop_loss')}，止盈：{original_strategy.get('take_profit')}

【审计报告】
{reviewer_report.get('full_report', '无')}
最大严重等级：{reviewer_report.get('max_severity', '无')}

【裁决规则】
- 若审计发现严重违规，必须推翻原策略。
- 若审计为存疑，可降仓执行或推翻。
- 若审计通过，维持原判。

【输出JSON】
{{
  "final_verdict": "维持原判/推翻",
  "final_direction": "long/short/neutral",
  "final_confidence": "高/中/低",
  "final_position_size": "重仓/中仓/轻仓/无",
  "entry_price_low": 0.0, "entry_price_high": 0.0,
  "stop_loss": 0.0, "take_profit": 0.0,
  "execution_plan": "",
  "risk_note": "",
  "audit_adopted": true,
  "audit_max_severity": "严重/中等/轻度/无"
}}
"""

# ------------------- 模型调用 -------------------
def call_trader(prompt: str) -> dict:
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=FAST_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=16384, timeout=TIMEOUT_SECONDS, stop=["}\n```"])
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip(): raise ValueError("空响应")
            json_str = extract_json(content)
            s = json.loads(json_str)
            # 方向标准化
            dir_map = {"做多":"long","做空":"short","观望":"neutral","long":"long","short":"short","neutral":"neutral"}
            s["direction"] = dir_map.get(s.get("direction",""), "neutral")
            pos_map = {"轻仓":"light","中仓":"medium","重仓":"heavy","无":"none"}
            s["position_size"] = pos_map.get(s.get("position_size",""), "none")
            conf_map = {"高":"high","中":"medium","低":"low"}
            s["confidence"] = conf_map.get(s.get("confidence",""), "medium")
            s.setdefault("reasoning",""); s.setdefault("risk_note",""); s["_model"] = resp.model
            return s
        except Exception as e:
            logger.warning(f"交易员调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {"direction":"neutral","confidence":"low","position_size":"none","entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,"execution_plan":"调用失败","reasoning":"","risk_note":"","cycle_phase":"","vote_result":{},"risk_reward_ratio":0,"_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

def call_reviewer(original_strategy: dict, data: dict, symbol: str) -> dict:
    prompt = build_reviewer_prompt(original_strategy, data, symbol)
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=FAST_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=4096, timeout=120)
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip(): raise ValueError("空响应")
            json_str = extract_json(content)
            s = json.loads(json_str)
            return {**s, "_model": resp.model}
        except Exception as e:
            logger.warning(f"审计官调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {"verdict":"通过","max_severity":"无","severity_counts":{},"full_report":"审计官调用失败","critical_items":[],"_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

def call_judge(original_strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    prompt = build_judge_prompt(original_strategy, reviewer_report, data, symbol)
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=TIMEOUT_SECONDS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=REASONING_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=16384, timeout=120)
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip(): raise ValueError("空响应")
            json_str = extract_json(content)
            s = json.loads(json_str)
            return {**s, "_model": resp.model}
        except Exception as e:
            logger.warning(f"委员会调用失败: {e}")
            if attempt == MAX_RETRIES-1:
                return {"final_verdict":"维持原判","final_direction":"neutral","final_confidence":"低","final_position_size":"无","entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,"execution_plan":"调用失败","risk_note":"","audit_adopted":False,"audit_max_severity":"无","_model":"fallback"}
            time.sleep(RETRY_BASE_WAIT**(attempt+1))

def apply_final_verdict(strategy: dict, judge: dict):
    final = judge.get("final_verdict","维持原判")
    logger.info(f"应用最终决议: {final}")
    if final == "推翻":
        direction = judge.get("final_direction","neutral")
        if direction == "neutral":
            _force_neutral(strategy, "委员会推翻并观望")
        else:
            strategy.update({k: judge[k] for k in ["direction","confidence","position_size","entry_price_low","entry_price_high","stop_loss","take_profit","execution_plan"] if k in judge})
            strategy["risk_note"] = judge.get("risk_note","")
    strategy["_judge_verdict"] = final
    return strategy