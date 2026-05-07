"""
deepseek.py — 最终版：七步数据确认 + 审计官核查清单 + 委员会独立裁决
取消微观数据时效约束，每步数据确认表防止遗漏，审计官逐条核查硬性标准，
委员会读取原始数据独立验证，每条裁决得出正确结论，输出可执行合约单。
"""

import os
import json
import time
import re
import math
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

# ---------- 标准化映射 ----------
def norm_direction(raw: str) -> str:
    if not raw:
        return "neutral"
    clean = raw.strip().lower()
    if clean in VALID_DIRECTIONS:
        return clean
    mapping = {"做多": "long", "做空": "short", "观望": "neutral"}
    return mapping.get(clean, "neutral")

def norm_confidence(raw: str) -> str:
    if not raw:
        return "medium"
    clean = raw.strip().lower()
    if clean in VALID_CONFIDENCES:
        return clean
    mapping = {"高": "high", "中": "medium", "低": "low"}
    return mapping.get(clean, "medium")

def norm_position_size(raw: str) -> str:
    if not raw:
        return "none"
    clean = raw.strip().lower()
    if clean in VALID_POSITION_SIZES:
        return clean
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
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ---------- 辅助 ----------
def _log_response(role: str, prompt: str, content: str, reasoning: str = None):
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/{role}_{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "content": content, "reasoning": reasoning}, f, ensure_ascii=False, indent=2)
    except:
        pass

def extract_json_safe(content: str) -> str:
    if not content or not content.strip():
        raise ValueError("空响应内容")
    m = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if m:
        json_str = m.group(1).strip()
        try:
            json.loads(json_str)
            return json_str
        except:
            pass
    m = re.search(r'```\s*([\s\S]*?)\s*```', content)
    if m:
        json_str = m.group(1).strip()
        try:
            json.loads(json_str)
            return json_str
        except:
            pass
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = content[start:end+1].strip()
        try:
            json.loads(json_str)
            return json_str
        except:
            pass
    if start != -1 and end != -1:
        json_str = content[start:end+1].strip()
        json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        try:
            json.loads(json_str)
            logger.warning("JSON通过转义修复成功")
            return json_str
        except:
            pass
    if json_str.startswith('{'):
        if not json_str.endswith('"') and not json_str.endswith('}'):
            json_str += '"}'
        elif json_str.endswith('"'):
            json_str += '}'
        try:
            json.loads(json_str)
            logger.warning("JSON通过暴力修补修复成功")
            return json_str
        except:
            pass
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
        atr_15m = data.get("atr_15m", 0)
        mark = data.get("mark_price", 0)
        ab_liq = data.get("above_liq", 0)
        bl_liq = data.get("below_liq", 0)
        bias_q = data.get("_bias_quality", "reliable")
        bias = data.get("direction_bias", 0.0)
        if (not ab_liq or ab_liq <= 0) and (not bl_liq or bl_liq <= 0) and direction != "neutral":
            _force_neutral(s, "清算数据缺失")
            return True, ""
        if atr_15m <= 0 or mark <= 0:
            if direction != "neutral":
                _force_neutral(s, "ATR或价格缺失")
                return True, ""
        if bias_q in ("reliable", "degraded") and abs(bias) > 0.4 and direction != "neutral":
            if (bias > 0 and direction == "short") or (bias < 0 and direction == "long"):
                _force_neutral(s, f"方向与锚点({bias:.3f})冲突")
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

# ---------- 清算穿刺 & 仪表盘 ----------
def compute_liquidation_bias(data: dict) -> dict:
    liq_r = data.get('liq_ratio', 1.0)
    cvd = data.get('cvd_slope', 0.0)
    taker = data.get('taker_ratio_1h', 0.5)
    ob_imb = data.get('orderbook_imbalance', 0.0)
    press = data.get('large_order_pressure', 0.0)
    pain = data.get('max_pain', 0.0)
    atr = data.get('atr', 0.0)
    mark = data.get('mark_price', 0.0)
    score = (liq_r - 1.0) * 0.4 + (1 if cvd > 0 else -1) * 0.3 + (taker - 0.5) * 0.3
    direction = 'balanced'
    if score > 0.15:
        direction = 'up'
    elif score < -0.15:
        direction = 'down'
    lure = (direction == 'up' and press < -0.5) or (direction == 'down' and press > 0.5)
    pain_eff = False
    if atr > 0 and pain > 0 and abs(pain - mark) < 1.0 * atr:
        if (direction == 'up' and pain > mark) or (direction == 'down' and pain < mark):
            pain_eff = True
    return {'puncture_direction': direction, 'puncture_score': score, 'lure_risk': lure, 'pain_magnet': pain_eff}

def build_expectation_dashboard(data: dict) -> str:
    basis_ann = data.get('basis_annualized', 0)
    basis_med = data.get('basis_median', 8)
    fund_pct = data.get('funding_percentile', 50)
    cgdi_pct = data.get('cgdi_percentile', 50)
    st_flow = data.get('stablecoin_trend_7d', 0)
    btc_dom = data.get('btc_dominance_trend_7d', 0)
    borrow = data.get('borrow_rate', 0) * 100
    pc = data.get('put_call_ratio', 1.0)
    price_pct = data.get('price_percentile', 50)
    vol_f = data.get('vol_factor', 1.0)
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
    'mark_price', 'atr', 'above_liq', 'below_liq', 'liq_ratio',
    'cvd_slope', 'taker_ratio_1h', 'oi_change_24h', 'funding_percentile',
    'orderbook_imbalance', 'large_order_pressure', 'max_pain', 'put_call_ratio',
    'basis_percentile', 'stablecoin_trend_7d', 'cgdi_percentile',
    'fear_greed', 'lth_realized_price', 'sth_realized_price', 'sth_sopr'
]

def compute_coverage(data: dict) -> dict:
    total = len(CORE_KEYS)
    available = sum(1 for k in CORE_KEYS if data.get(k) is not None)
    coverage = available / total if total > 0 else 0.0
    return {"available": available, "total": total, "coverage": coverage}

# ---------- 首席交易员 Prompt ----------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    if cross_symbol is None:
        cross_symbol = "ETH" if symbol == "BTC" else "BTC"
    coverage = compute_coverage(data)

    def safe_val(key, default=0.0, scale=1.0, fmt=".2f"):
        raw = data.get(key)
        if raw is None:
            return ("[N/A]", True)
        try:
            val = float(raw) * scale
        except:
            return ("[N/A]", True)
        if math.isnan(val) or math.isinf(val):
            return ("[N/A]", True)
        try:
            return (f"{val:{fmt}}", False)
        except:
            return ("[N/A]", True)

    # 提取所有字段（省略重复，实际完整）
    mark_str, _ = safe_val('mark_price', fmt=".2f")
    atr_str, _ = safe_val('atr', fmt=".2f")
    fear_greed = data.get('fear_greed', 50)
    lth_str, _ = safe_val('lth_realized_price', fmt=".2f")
    sth_str, _ = safe_val('sth_realized_price', fmt=".2f")
    sopr_str, _ = safe_val('sth_sopr', 1.0, fmt=".3f")
    stable_str, _ = safe_val('stablecoin_trend_7d', fmt="+.1f")
    oi_chg_str, _ = safe_val('oi_change_24h', fmt="+.1f")
    fund_pct_str, _ = safe_val('funding_percentile', 50, fmt=".0f")
    cvd_str, _ = safe_val('cvd_slope', fmt=".4f")
    taker_str, _ = safe_val('taker_ratio_1h', fmt=".3f")
    nf24h_str, _ = safe_val('netflow_24h', scale=1/1e6, fmt=".1f")
    abv_liq_str, _ = safe_val('above_liq', scale=1/1e9, fmt=".2f")
    blw_liq_str, _ = safe_val('below_liq', scale=1/1e9, fmt=".2f")
    liq_r_str, _ = safe_val('liq_ratio', fmt=".2f")
    abv_trig = data.get('above_trigger', 'N/A')
    blw_trig = data.get('below_trigger', 'N/A')
    lgs_str, _ = safe_val('large_sell_value', scale=1/1e6, fmt=".1f")
    lgb_str, _ = safe_val('large_buy_value', scale=1/1e6, fmt=".1f")
    press_str, _ = safe_val('large_order_pressure', fmt=".3f")
    ob_imb_str, _ = safe_val('orderbook_imbalance', fmt=".3f")
    lure_str, _ = safe_val('lure_risk_factor', fmt=".2f")
    pain_str, _ = safe_val('max_pain', fmt=".2f")
    pc_str, _ = safe_val('put_call_ratio', fmt=".4f")
    basis_pct_str, _ = safe_val('basis_percentile', 50, fmt=".0f")
    btc_dom_str, _ = safe_val('btc_dominance_trend_7d', fmt="+.1f")
    borrow_str, _ = safe_val('borrow_rate', scale=100, fmt=".2f")
    exch_str, _ = safe_val('exchange_btc_change_24h', fmt="+.0f")
    spot24_str, _ = safe_val('spot_netflow_24h', scale=1/1e6, fmt=".1f")
    spot_div_str, _ = safe_val('spot_vs_futures_divergence', fmt=".2f")
    top_ls_str, _ = safe_val('top_ls_percentile', 50, fmt=".0f")
    price_pct_str, _ = safe_val('price_percentile', 50, fmt=".0f")
    vol_f_str, _ = safe_val('vol_factor', 1.0, fmt=".2f")
    cgdi_pct_str, _ = safe_val('cgdi_percentile', 50, fmt=".0f")
    direction_bias = data.get('direction_bias', 0.0)
    bias_quality = data.get('_bias_quality', 'reliable')
    puncture = compute_liquidation_bias(data)
    dashboard = build_expectation_dashboard(data)

    # 数据质量标记
    data_quality_map = data.get("data_quality", {})
    liq_quality = "低" if data_quality_map.get("heatmap") == "❌ 缺失" else "高"
    exch_quality = "低" if data_quality_map.get("exchange_btc") == "❌ 缺失" else "高"
    onchain_quality = "低" if data_quality_map.get("sth_sopr") == "❌ 缺失" else "高"

    # 快速响应因子（基于现有数据粗略估计）
    price_24h_pct = data.get('price_percentile', 50) / 100.0
    vol_surge = 1.0
    atr_ratio = (data.get('mark_price', 0) * 0.02 / data.get('atr', 1.0)) if data.get('atr', 0) > 0 else 1.0

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

    # 七步分析框架文本
    SEVEN_STEP_FRAMEWORK = """
## 步骤1：数据全景与可信度评估
**数据确认**（逐项填写）：
| 数据项 | 当前值 | 这意味着什么？ |
|--------|--------|---------------|
| 清算图质量 | {liq_quality} | |
| 链上数据质量 | {onchain_quality} | |
| 交易所数据质量 | {exch_quality} | |
| 核心数据覆盖率 | {coverage}% | |
| 方向锚点可信度 | {bias_quality} | |
- 任何“低”标记的数据源，关联指标权重减半。
- 若≥2个数据源为“低”，confidence强制=“低”，position_size强制=“轻仓/无”。
**步骤1结论**：整体数据质量[高/中/低]，数据源异常[0/1/2/3]个，仓位上限[重仓/中仓/轻仓/无]。

## 步骤2：宏观结构与链上底色
**数据确认**（逐项填写）：
| 数据项 | 当前值 | 这意味着什么？ |
|--------|--------|---------------|
| 现价 | {mark_str} | |
| LTH成本 | {lth_str} | |
| STH成本 | {sth_str} | |
| STH SOPR | {sopr_str} | |
| LTH SOPR | {data.get('lth_sopr', 1.0):.3f} | |
| 稳定币市值趋势 | {stable_str}% | |
| 交易所BTC余额变化 | {exch_str} BTC | |
| 借贷利率 | {borrow_str}% | |
| 恐慌贪婪指数 | {fear_greed} | |
| CGDI绝对值 | {data.get('cgdi_current',0):.0f} | |
| 稳定币市值 | {data.get('stablecoin_mcap',0)/1e9:.1f}B | |
| BTC市值占比趋势 | {btc_dom_str}% | |
完成因果模板：“当前市场处于【牛/熊/震荡】的【早期/中期/晚期】，核心证据是【LTH/STH成本与现价关系】揭示了【持股盈亏状态】，结合【稳定币/交易所存量】变动，表明资金是【流入/流出】。结构性支撑带在【】，宏观象限标签为【趋势开端/趋势中继/转折点/震荡无序】。”
**步骤2结论**：宏观象限[趋势开端/趋势中继/转折点/震荡无序]，结构性偏向[做多/做空/中性]。

## 步骤3：多空动能正交分解
**数据确认**（逐项填写，为每项打分-2~+2）：
| 数据项 | 当前值 | 得分 | 独立性 |
|--------|--------|------|--------|
| 上方清算簇强度 | {abv_liq_str}B, 距{abv_trig}点 | | |
| 下方清算簇强度 | {blw_liq_str}B, 距{blw_trig}点 | | |
| CVD斜率 | {cvd_str} | | |
| CVD加速度 | {data.get('cvd_acceleration',0):.4f} | | |
| CVD均值 | {data.get('cvd_mean',0):.2f}M | | |
| 主动买卖比(1h) | {taker_str} | | |
| 大单压迫比 | {press_str} | | |
| 大额卖单/买单 | {lgs_str}M/{lgb_str}M | | |
| OI 24h变化 | {oi_chg_str}% | | |
| OI加速度 | {data.get('oi_acceleration',0):.4f} | | |
| 订单簿失衡 | {ob_imb_str} | | |
| 全市场OI变化 | {data.get('agg_oi_change_24h',0):+.1f}% | | |
| 修正项 | 当前值 | 修正分 |
|--------|--------|--------|
| 全球多空比 | {data.get('global_ls_ratio',1):.2f} | |
| 价格7日分位 | {price_pct_str}% | |
| 散户/鲸鱼背离 | {data.get('retail_whale_divergence',0):.3f} | |
| 大户多空比分位 | {top_ls_str}% | |
无独立性的因子贡献×0.6。全球多空比>2且价格7日分位>80%时，修正系数-1。
**步骤3结论**：动能净得分[+X.XX]，净倾向[强多/偏多/中性/偏空/强空]。

## 步骤4：流动性猎杀博弈
**数据确认**（逐项填写）：
| 数据项 | 当前值 | 这意味着什么？ |
|--------|--------|---------------|
| 清算比值(下/上) | {liq_r_str} | |
| ATR(4h) | {atr_str} | |
| 诱饵风险 | {lure_str} | |
| 期权最大痛点 | {pain_str} | |
| 期权磁吸 | {puncture['pain_magnet']} | |
| 1h多头爆仓 | {data.get('long_liq_1h',0):.2f}M | |
| 1h空头爆仓 | {data.get('short_liq_1h',0):.2f}M | |
| 爆仓偏空比 | {data.get('liq_bias_1h',0):.3f} | |
| 5分钟净流 | {data.get('netflow_5m',0)/1e6:.1f}M | |
| 1小时净流 | {data.get('netflow_1h',0)/1e6:.1f}M | |
推演两个方向：路径A（现价 ± 1.5 ATR），路径B（反向穿刺诱饵）。
**步骤4结论**：最可能猎杀方向[向上/向下/均衡]，确定性[高/中/低]。

## 步骤5：预期差仪表盘解读
**数据确认**（逐项填写）：
| 数据项 | 当前值 | 市场定价了什么？ |
|--------|--------|-----------------|
| 恐慌贪婪 vs 稳定币趋势 | {fear_greed} vs {stable_str}% | |
| 资金费率分位 | {fund_pct_str}% | |
| 资金费率动量 | {data.get('funding_momentum',0):.6f} | |
| P/C比 | {pc_str} | |
| 合约基差分位 | {basis_pct_str}% | |
| 合约基差当前值 | {data.get('basis_current',0):.4f} | |
| 现货/期货背离度 | {spot_div_str} | |
| 现货1h净流 | {data.get('spot_netflow_1h',0)/1e6:.1f}M | |
| 现货24h净流 | {spot24_str}M | |
| 期货24h净流 | {nf24h_str}M | |
| CGDI分位 | {cgdi_pct_str}% | |
| 价格7日分位 | {price_pct_str}% | |
| 波动因子 | {vol_f_str} | |
寻找周期一致的矛盾指标对，推演预期差。
**步骤5结论**：预期差方向[向上/向下/无显著预期差]，潜在意外方[多头/空头/无]。

## 步骤6：跨币种生态验证
**数据确认**（逐项填写）：
| 数据项 | BTC | ETH | 方向一致？ |
|--------|-----|-----|-----------|
| 清算比值 | {data.get('liq_ratio',1):.2f} | {eth_data.get('liq_ratio',0):.2f} | |
| CVD斜率 | {data.get('cvd_slope',0):.4f} | {eth_data.get('cvd_slope',0):.4f} | |
| OI 24h变化 | {data.get('oi_change_24h',0):+.1f}% | {eth_data.get('oi_change_24h',0):+.1f}% | |
| 顶多空分位 | {data.get('top_ls_percentile',50):.0f}% | {eth_data.get('top_ls_percentile',50):.0f}% | |
| 爆仓偏空比 | {data.get('liq_bias_1h',0):.3f} | {eth_data.get('liq_bias_1h',0):.3f} | |
| 资金费率分位 | {fund_pct_str}% | {eth_data.get('funding_percentile',50):.0f}% | |
| 期权痛点 | {pain_str} | {eth_data.get('max_pain',0):.2f} | |
| ETH/BTC汇率分位 | — | {data.get('eth_btc_percentile',50):.0f}% | |
输出仅允许：“BTC与ETH呈强协同，支持同向交易” 或 “BTC与ETH显著背离，方向性押注风险上升”。cross_coin_action字段仅允许：【同向可做】【背离警告】【无明确信号】。
**步骤6结论**：跨币种信号[同向可做/背离警告/无明确信号]。

## 步骤7：策略生成与反向压力测试
**最终确认**（逐项填写）：
| 数据项 | 当前值 | 对策略的影响 |
|--------|--------|-------------|
| 价格24h分位 | {price_24h_pct:.2f} | |
| 成交量爆发比 | {vol_surge:.1f}x | |
| ATR振幅比 | {atr_ratio:.1f} | |
| 方向锚点 | {direction_bias:.3f} | |
| 宏观象限标签 | （步骤2结论） | |
| 动能净得分 | （步骤3结论） | |
| 猎杀方向 | （步骤4结论） | |
| 预期差方向 | （步骤5结论） | |
| 跨币种信号 | （步骤6结论） | |
完成紧急响应检查、动态权重分配、反向压力测试。
**步骤7结论**：最终方向[做多/做空/观望]，仓位[重仓/中仓/轻仓/无]，置信度[高/中/低]。
"""

    prompt = f"""你是一位拥有 15 年实战经验、以量化严谨著称的加密货币首席交易员。你的任务是结构化解构市场，而非提供交易建议。你完全信任外部数据管道的质量标记，并严格执行硬约束。

严格按照「七步递进分析框架」输出，每步必须先完成数据确认表再定性分析，每步必须有**结论**。总字数 ≤ 3000 字。

【数据与锚点】
覆盖率：{coverage['coverage']:.0%}（{coverage['available']}/{coverage['total']}）
方向锚点 direction_bias = {direction_bias:.3f}，可信度：{bias_quality}
清算穿刺预判方向：{puncture['puncture_direction']}，得分：{puncture['puncture_score']:.2f}
诱饵风险：{puncture['lure_risk']}，期权磁吸：{puncture['pain_magnet']}

【外部数据管道质量标记 - 你只可读取，不可修改】
- 清算图质量：{liq_quality}
- 链上数据质量：{onchain_quality}
- 交易所数据质量：{exch_quality}

【市场快速响应因子 - 用于检测急转弯状态】
- 价格24h分位：{price_24h_pct:.2f}
- 成交量爆发比：{vol_surge:.1f}x
- ATR振幅比：{atr_ratio:.1f}

{dashboard}

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

{cross_context}

---
{SEVEN_STEP_FRAMEWORK}

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
  "reasoning": "每步以【步骤X：名称】开头，包含数据确认表和结论行",
  "risk_note": "必须包含数据质量约束声明及反向情景退出条件",
  "risk_reward_ratio": 0.0,
  "data_quality_constraints": "列出步骤1中所有被激活的硬约束",
  "emergency_mode": "是/否",
  "cross_coin_action": "同向可做/背离警告/无明确信号",
  "step_conclusions": {{"步骤1": "", "步骤2": "", "步骤3": "", "步骤4": "", "步骤5": "", "步骤6": "", "步骤7": ""}},
  "vote_result": {{"宏观底色": "", "多空动能": "", "博弈维度": "", "预期差": "", "跨币种": "", "最终方向": ""}}
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
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content or ""
            _log_response("trader", prompt, content)
            if not content.strip():
                raise ValueError("空响应")
            json_str = extract_json_safe(content)
            s = json.loads(json_str)
            s["direction"] = norm_direction(s.get("direction", ""))
            s["position_size"] = norm_position_size(s.get("position_size", ""))
            s["confidence"] = norm_confidence(s.get("confidence", ""))
            s.setdefault("reasoning", "")
            s.setdefault("risk_note", "")
            s.setdefault("execution_plan", "")
            s["reasoning"] = format_reasoning(s["reasoning"])
            s["_model_used"] = resp.model
            return s
        except Exception as e:
            logger.warning(f"交易员调用失败: {e}")
            if attempt == MAX_RETRIES - 1:
                return {"direction": "neutral", "confidence": "low", "position_size": "none", "entry_price_low": 0, "entry_price_high": 0, "stop_loss": 0, "take_profit": 0, "execution_plan": "调用失败", "reasoning": "调用失败", "risk_note": "", "_model_used": "fallback"}
            time.sleep(RETRY_BASE_WAIT ** (attempt + 1))


# ------------------- 审计官（核查清单版） -------------------
def call_reviewer(strategy: dict, data: dict, symbol: str) -> dict:
    direction_bias = data.get('direction_bias', 0.0)
    coverage_info = compute_coverage(data)
    puncture = compute_liquidation_bias(data)
    bias_quality = data.get('_bias_quality', 'reliable')

    prompt = f"""你是一位独立的风险审计官，对首席交易员的七步分析进行逐步骤严格审查。你必须基于硬性标准输出结构化报告。

【审计背景】
标的：{symbol}，锚点：{direction_bias:.3f}，可信度：{bias_quality}
覆盖率：{coverage_info['coverage']:.0%}，穿刺预判：{puncture['puncture_direction']}
交易员方向：{strategy.get('direction')}，仓位：{strategy.get('position_size')}

交易员推演原文：
{format_reasoning(strategy.get('reasoning', '无'))}

你必须逐条回答以下核查清单，每一条只能回答“✅通过”或“❌违规（具体说明）”。禁止泛泛而谈。

【核查清单】
1. 步骤1是否完整填写了数据确认表？（逐项核对，缺失一项即为违规）
2. 步骤2是否完成了因果模板？是否引用了所有指定数据？
3. 步骤3的动能打分是否引用了具体数值？（若只有分数没有数值，违规）
4. 步骤4是否推演了路径A和路径B两个方向？
5. 步骤5是否找到了至少一对矛盾指标？（若声称“没有矛盾”但实际存在明显矛盾，违规）
6. 步骤6是否引用了跨币种数据？
7. 步骤7是否计算了盈亏比？（未计算或为零，违规）
8. 多空博弈中，多空双方是否各列出了至少3条论据？
9. 交叉质询是否完成？（双方各攻击了对方的一条论据）
10. 是否有数据被错误引用？（例如把CVD斜率-32939 写成 +32939）

同时，你必须指出交易员推演中最致命的缺陷（若存在），以及未被处理的关键反证信号。

输出JSON格式：
{{
  "step_audits": [
    {{"step": 1, "verdict": "合格/存在瑕疵/严重错误", "issues": [{{"type": "数据遗漏/误读/逻辑矛盾/反证缺失", "description": "...", "severity": "高/中/低", "evidence": "..."}}]}},
    ...
  ],
  "overall_verdict": "通过/存疑/驳回",
  "max_severity": "严重/中等/轻度/无",
  "severity_summary": {{"严重": 0, "中等": 0, "轻度": 0}},
  "checklist": [
    {{"item": 1, "result": "✅通过/❌违规", "detail": ""}},
    ...
  ],
  "most_fatal_flaw": "最致命的缺陷（一句话）",
  "unhandled_contrarian_signal": "未被处理的关键反证信号",
  "full_report": "完整审计文本"
}}
"""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=120)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                timeout=120,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content or ""
            _log_response("reviewer", prompt, content)
            rev = json.loads(extract_json_safe(content))
            if not rev.get("full_report"):
                rev["full_report"] = content
            rev["full_report"] = format_reasoning(rev["full_report"])
            if sum(rev.get("severity_summary", {}).values()) == 0 and rev.get("overall_verdict") == "驳回":
                rev["severity_summary"]["严重"] = 1
                rev["max_severity"] = "严重"
            return {**rev, "_model": resp.model}
        except Exception as e:
            logger.warning(f"审计官调用失败: {e}")
            if attempt == MAX_RETRIES - 1:
                return {"overall_verdict": "驳回", "max_severity": "严重", "severity_summary": {"严重": 1, "中等": 0, "轻度": 0}, "step_audits": [], "checklist": [], "most_fatal_flaw": "审计失败", "unhandled_contrarian_signal": "", "full_report": "审计失败", "_model": "fallback"}
            time.sleep(RETRY_BASE_WAIT ** (attempt + 1))


# ------------------- 交易委员会（独立裁决版） -------------------
def call_judge(strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    direction_bias = data.get('direction_bias', 0.0)

    # 构建审计指控摘要
    audit_charges = ""
    for audit in reviewer_report.get("step_audits", []):
        for issue in audit.get("issues", []):
            if issue.get("severity") in ("高", "中"):
                audit_charges += f"步骤{audit['step']}: {issue['description']} (严重性:{issue['severity']})\n"

    # 将完整市场数据提供给委员会
    market_data_summary = f"""
现价：{data.get('mark_price',0):.2f}，ATR：{data.get('atr',0):.2f}，恐慌贪婪：{data.get('fear_greed',50)}
LTH成本：{data.get('lth_realized_price',0):.2f}，STH成本：{data.get('sth_realized_price',0):.2f}，STH SOPR：{data.get('sth_sopr',1):.3f}
CVD斜率：{data.get('cvd_slope',0):.4f}，主动买卖比：{data.get('taker_ratio_1h',1):.3f}
24h期货净流：{data.get('netflow_24h',0)/1e6:.1f}M，现货24h净流：{data.get('spot_netflow_24h',0)/1e6:.1f}M
清算比值：{data.get('liq_ratio',1):.2f}，上方触发距：{data.get('above_trigger','N/A')}点，下方触发距：{data.get('below_trigger','N/A')}点
大单压迫比：{data.get('large_order_pressure',0):.3f}，订单簿失衡：{data.get('orderbook_imbalance',0):.3f}
期权痛点：{data.get('max_pain',0):.2f}，P/C比：{data.get('put_call_ratio',1):.4f}
OI 24h变化：{data.get('oi_change_24h',0):+.1f}%，资金费率分位：{data.get('funding_percentile',50):.0f}%
方向锚点：{direction_bias:.3f}，可信度：{data.get('_bias_quality','reliable')}
"""

    prompt = f"""你是交易委员会主席，拥有最终决策权。你必须基于原始市场数据进行独立验证，逐条裁决审计指控，并给出正确的最终结论。

【市场数据】（与交易员分析时完全一致）
{market_data_summary}

【交易员策略】
方向：{strategy.get('direction')}，仓位：{strategy.get('position_size')}，置信度：{strategy.get('confidence')}
入场：{strategy.get('entry_price_low')}-{strategy.get('entry_price_high')}
止损：{strategy.get('stop_loss')}，止盈：{strategy.get('take_profit')}
推演：{format_reasoning(strategy.get('reasoning', '无'))}

【审计报告】
{format_reasoning(reviewer_report.get('full_report', '无'))}
审计结论：{reviewer_report.get('overall_verdict')}，最高严重性：{reviewer_report.get('max_severity')}
最致命缺陷：{reviewer_report.get('most_fatal_flaw', '无')}
审计指控：
{audit_charges if audit_charges else "无严重指控"}

你需要：
1. 逐条审计指控，读取原始数据进行独立验证。对于每条指控：
   - 裁决该指控是成立还是不成立
   - 若成立，必须给出纠正后的正确结论（例如：正确的动能得分应为XX，正确的盈亏比应为XX）
   - 若不成立，说明驳回理由
2. 基于原始数据，发现交易员和审计官都未提及的重要信号（若有）。
3. 综合所有裁决，制定一份可立即执行的最终合约策略，包括方向、仓位、入场、止损、止盈、盈亏比、执行指令。

输出JSON格式：
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
    {{"step": 1, "issue": "指控内容", "adopted": true/false, "correct_conclusion": "若成立，给出正确结论", "reason": "裁决依据"}}
  ],
  "independent_findings": "委员会独立发现的重要信号（若无写'无'）",
  "final_reasoning": "最终裁决理由，必须包含对关键指控的回应和最终策略的制定逻辑"
}}
"""
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=120)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=REASONING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                timeout=120,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content or ""
            _log_response("judge", prompt, content)
            result = json.loads(extract_json_safe(content))
            result["final_direction"] = norm_direction(result.get("final_direction", ""))
            result["final_position_size"] = norm_position_size(result.get("final_position_size", ""))
            result["final_confidence"] = norm_confidence(result.get("final_confidence", ""))
            if result.get("final_verdict") == "维持原判":
                for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit", "execution_plan", "risk_note"]:
                    if not result.get(f):
                        result[f] = strategy.get(f, 0)
            result["final_reasoning"] = format_reasoning(result.get("final_reasoning", ""))
            return {**result, "_model": resp.model}
        except Exception as e:
            logger.warning(f"委员会调用失败: {e}")
            if attempt == MAX_RETRIES - 1:
                return {"final_verdict": "推翻", "final_direction": "neutral", "final_confidence": "low", "final_position_size": "none", "entry_price_low": 0, "entry_price_high": 0, "stop_loss": 0, "take_profit": 0, "execution_plan": "失败", "risk_note": "", "final_reasoning": "失败", "_model": "fallback"}
            time.sleep(RETRY_BASE_WAIT ** (attempt + 1))


def apply_final_verdict(strategy: dict, judge_result: dict) -> dict:
    verdict = judge_result.get("final_verdict", "维持原判")
    logger.info(f"应用最终决议: {verdict}")
    strategy["_judge_verdict"] = verdict
    strategy["_judge_reasoning"] = judge_result.get("final_reasoning", "")
    fields = ["direction", "confidence", "position_size", "entry_price_low", "entry_price_high", "stop_loss", "take_profit", "execution_plan", "risk_note"]
    if verdict in ("推翻", "修改执行"):
        if judge_result.get("final_direction") == "neutral":
            _force_neutral(strategy, "委员会改为观望")
        else:
            for k in fields:
                if k in judge_result and judge_result[k] is not None:
                    strategy[k] = judge_result[k]
    return strategy
