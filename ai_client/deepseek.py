"""
deepseek.py — 完整修正版
- 透明化缺失数据，AI 可见“缺失”而非虚假默认值
- 修正数据缺失时约束提示
- 增强策略校验：致命缺失直接 neutral
- 清理 call_judge 冗余代码
- 需要同步修改 CoinGlassClient._build_main_data 中的稳定币、BTC占比、借贷利率提取（见末尾注释）
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

# 模型定义
FAST_MODEL = "deepseek-v4-pro"
REASONING_MODEL = "deepseek-v4-pro"


# ---------- 辅助函数 ----------
def _safe_float(pattern, text, default=0.0):
    """从文本中提取浮点数"""
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return default
    return default


def _log_response(prompt: str, content: str, reasoning: str = None):
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/deepseek_{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "content": content, "reasoning": reasoning}, f, ensure_ascii=False, indent=2)
    except:
        pass


def extract_json(content: str) -> str:
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
    raise ValueError("JSON 未闭合")


def round_to_tick(price: float) -> float:
    return round(price / TICK_SIZE) * TICK_SIZE


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
    if direction not in ["long", "short", "neutral"]:
        return False, f"无效方向: {direction}"

    if data:
        atr_15m = data.get("atr_15m", 0)
        above_liq = data.get("above_liq", 0)
        below_liq = data.get("below_liq", 0)
        mark_price = data.get("mark_price", 0)
        direction_bias = data.get("direction_bias", 0.0)

        # 致命缺失 → neutral
        if (not below_liq or below_liq <= 0) and (not above_liq or above_liq <= 0) and direction != "neutral":
            _force_neutral(s, "清算数据缺失，强制 neutral")
            return True, "已自动修正为观望"
        if atr_15m <= 0 or mark_price <= 0:
            if direction != "neutral":
                _force_neutral(s, "ATR 或价格缺失，强制 neutral")
                return True, "已自动修正为观望"

        # 置信度降级
        if atr_15m <= 0 and s.get("confidence") == "high":
            s["confidence"] = "medium"
            logger.warning("核心数据缺失(atr_15m)，置信度强制降级为 medium")
        if data.get("cvd_slope") is None and s.get("confidence") == "high":
            s["confidence"] = "medium"
            logger.warning("核心数据缺失(cvd_slope)，置信度强制降级为 medium")

        # direction_bias 强冲突降级
        if abs(direction_bias) > 0.4 and direction != "neutral":
            if (direction_bias > 0 and direction == "short") or (direction_bias < 0 and direction == "long"):
                if s.get("confidence") == "high":
                    s["confidence"] = "medium"
                    logger.warning(f"方向与 direction_bias({direction_bias:.3f})冲突，置信度降级为 medium")

    if direction == "neutral":
        for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
            if s.get(f, 0) != 0:
                return False, f"neutral 信号不应有非零的 {f}"
        return True, ""

    for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
        val = s.get(f)
        if val is None or float(val) <= 0:
            return False, f"缺少或无效的 {f}"

    entry_low = float(s["entry_price_low"])
    entry_high = float(s["entry_price_high"])
    stop_loss = float(s["stop_loss"])
    take_profit = float(s["take_profit"])

    if entry_low > entry_high:
        logger.warning(f"入场区间下限({entry_low})大于上限({entry_high})，已自动交换并将仓位降为light")
        s["entry_price_low"], s["entry_price_high"] = entry_high, entry_low
        s["position_size"] = "light"

    if direction == "long" and stop_loss >= entry_low:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间下方，请人工确认。"
    elif direction == "short" and stop_loss <= entry_high:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间上方，请人工确认。"

    try:
        if direction == "long":
            risk = entry_high - stop_loss
            reward = take_profit - entry_high
        else:
            risk = stop_loss - entry_low
            reward = entry_low - take_profit
        if risk > 0:
            s["_calculated_rr"] = round(reward / risk, 2)
        else:
            s["_calculated_rr"] = 0.0
    except Exception as e:
        logger.warning(f"计算盈亏比时出错: {e}")
        s["_calculated_rr"] = None

    return True, ""


# ------------------- 构建策略提示词 -------------------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    if cross_symbol is None:
        cross_symbol = "ETH" if symbol == "BTC" else "BTC"

    # ---------- 辅助：安全取值并标记缺失 ----------
    def safe_val(key, default=0.0, scale=1.0, fmt=".2f", null_val=None):
        """返回 (显示字符串, 是否为缺失)"""
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
        cross_context = f"""
【{cross_symbol} 跨币种数据 - 仅用于第六步】
现价：{cross_mark:.2f}
清算：上方{eth_data.get('above_liq', 0)/1e9:.2f}B / 下方{eth_data.get('below_liq', 0)/1e9:.2f}B (比值{eth_data.get('liq_ratio', 0):.2f})
OI分位：{eth_data.get('oi_percentile', 50):.0f}%　OI 24h变化：{eth_data.get('oi_change_24h', 0):+.1f}%
费率分位：{eth_data.get('funding_percentile', 50):.0f}%　费率当前：{eth_data.get('funding_rate', 0):.4f}%
顶多空分位：{eth_data.get('top_ls_percentile', 50):.0f}%
CVD斜率：{eth_data.get('cvd_slope', 0):.4f}
爆仓偏空比：{eth_data.get('liq_bias_1h', 0):.3f}
期权：P/C比{eth_data.get('put_call_ratio', 0):.4f}　最大痛点{eth_data.get('max_pain', 0):.2f}
"""
        if not eth_data.get("_complete", False):
            cross_context += "\n⚠️ 以上部分数据可能缺失，请基于可用数据完成跨币种验证。"
    else:
        cross_context = "⚠️ 无跨币种数据"

    # ---------- 提取所有显示字段 ----------
    mark_price_str, _ = safe_val('mark_price', fmt=".2f")
    price_percentile_str, _ = safe_val('price_percentile', 50.0, fmt=".0f")
    atr_str, atr_miss = safe_val('atr', fmt=".2f")
    atr_1h_ratio_str, _ = safe_val('atr_1h_ratio', fmt=".1f")
    vol_factor_str, _ = safe_val('vol_factor', 1.0, fmt=".2f")
    cgdi_str, _ = safe_val('cgdi_current', fmt=".0f")
    cgdi_perc_str, _ = safe_val('cgdi_percentile', 50.0, fmt=".0f")
    fear_greed = data.get('fear_greed', 50)
    fear_greed_prev = data.get('fear_greed_prev_7d', 50)

    above_liq_str, _ = safe_val('above_liq', scale=1/1e9, fmt=".2f")
    below_liq_str, _ = safe_val('below_liq', scale=1/1e9, fmt=".2f")
    liq_ratio_str, _ = safe_val('liq_ratio', fmt=".2f")
    above_cluster = data.get('above_cluster', 'N/A')
    above_trigger = data.get('above_trigger', 'N/A')
    below_cluster = data.get('below_cluster', 'N/A')
    below_trigger = data.get('below_trigger', 'N/A')

    large_sell_str, _ = safe_val('large_sell_value', scale=1/1e6, fmt=".1f")
    large_buy_str, _ = safe_val('large_buy_value', scale=1/1e6, fmt=".1f")
    pressure_str, _ = safe_val('large_order_pressure', fmt=".3f")
    ob_bids_str, _ = safe_val('orderbook_bids', scale=1/1e6, fmt=".1f")
    ob_asks_str, _ = safe_val('orderbook_asks', scale=1/1e6, fmt=".1f")
    ob_imbalance_str, _ = safe_val('orderbook_imbalance', fmt=".3f")
    lure_str, _ = safe_val('lure_risk_factor', fmt=".2f")

    oi_str, _ = safe_val('oi', scale=1/1e9, fmt=".2f")
    oi_perc_str, _ = safe_val('oi_percentile', 50.0, fmt=".0f")
    oi_chg_str, _ = safe_val('oi_change_24h', fmt="+.1f")
    agg_oi_chg_str, _ = safe_val('agg_oi_change_24h', fmt="+.1f")
    fund_rate_str, _ = safe_val('funding_rate', fmt=".4f")
    fund_perc_str, _ = safe_val('funding_percentile', 50.0, fmt=".0f")
    fund_mom_str, _ = safe_val('funding_momentum', fmt=".6f")

    top_ls_str, _ = safe_val('top_ls_ratio', fmt=".2f")
    top_ls_perc_str, _ = safe_val('top_ls_percentile', 50.0, fmt=".0f")
    global_ls_str, _ = safe_val('global_ls_ratio', fmt=".2f")
    div_str, _ = safe_val('retail_whale_divergence', fmt=".3f")

    long_liq_str, _ = safe_val('long_liq_1h', scale=1/1e6, fmt=".1f")
    short_liq_str, _ = safe_val('short_liq_1h', scale=1/1e6, fmt=".1f")
    liq_bias_str, _ = safe_val('liq_bias_1h', fmt=".3f")

    cvd_slope_str, _ = safe_val('cvd_slope', fmt=".4f")
    cvd_accel_str, _ = safe_val('cvd_acceleration', fmt=".4f")
    cvd_mean_str, _ = safe_val('cvd_mean', fmt=".2f")
    taker_str, _ = safe_val('taker_ratio_1h', fmt=".3f")

    nf5m_str, _ = safe_val('netflow_5m', scale=1/1e6, fmt=".1f")
    nf1h_str, _ = safe_val('netflow_1h', scale=1/1e6, fmt=".1f")
    nf24h_str, _ = safe_val('netflow_24h', scale=1/1e6, fmt=".1f")
    exch_btc_str, _ = safe_val('exchange_btc_change_24h', fmt="+.0f")

    spot_1h_str, _ = safe_val('spot_netflow_1h', scale=1/1e6, fmt=".1f")
    spot_24h_str, _ = safe_val('spot_netflow_24h', scale=1/1e6, fmt=".1f")
    spot_div_str, _ = safe_val('spot_vs_futures_divergence', fmt=".2f")

    max_pain_str, _ = safe_val('max_pain', fmt=".2f")
    pc_str, _ = safe_val('put_call_ratio', fmt=".4f")

    eth_btc_str, _ = safe_val('eth_btc_ratio', fmt=".4f")
    eth_btc_ma_str, _ = safe_val('eth_btc_ma_7d', fmt=".4f")
    eth_btc_perc_str, _ = safe_val('eth_btc_percentile', 50.0, fmt=".0f")

    basis_str, _ = safe_val('basis_current', fmt=".4f")
    basis_perc_str, _ = safe_val('basis_percentile', 50.0, fmt=".0f")

    stable_mcap_str, _ = safe_val('stablecoin_mcap', scale=1/1e9, fmt=".2f")
    stable_trend_str, _ = safe_val('stablecoin_trend_7d', fmt="+.1f")
    btc_dom_str, _ = safe_val('btc_dominance', fmt=".1f")
    btc_dom_trend_str, _ = safe_val('btc_dominance_trend_7d', fmt="+.1f")

    lth_rp_str, _ = safe_val('lth_realized_price', fmt=".2f")
    sth_rp_str, _ = safe_val('sth_realized_price', fmt=".2f")
    lth_sopr_str, _ = safe_val('lth_sopr', 1.0, fmt=".3f")
    sth_sopr_str, _ = safe_val('sth_sopr', 1.0, fmt=".3f")
    borrow_str, _ = safe_val('borrow_rate', scale=100, fmt=".2f")

    direction_bias = data.get('direction_bias', 0.0)

    # ---------- 缺失约束 ----------
    data_quality = data.get("data_quality", {})
    core_missing = []
    if data_quality.get("kline") == "❌ 缺失":
        core_missing.extend(["atr_15m", "price_percentile", "mark_price"])
    if data_quality.get("heatmap") == "❌ 缺失":
        core_missing.extend(["above_liq", "below_liq", "liq_ratio"])
    if data_quality.get("cvd") == "❌ 缺失":
        core_missing.append("cvd_slope")
    if atr_miss:
        core_missing.append("atr_15m")
    constraint_note = ""
    if core_missing:
        unique_missing = list(set(core_missing))
        constraint_note = f"\n【重要约束】以下核心数据缺失：{', '.join(unique_missing)}。你必须将最终置信度设为 'low'；若清算数据缺失，则必须输出 'neutral'。\n"

    # ---------- 拼接完整 prompt ----------
    prompt = f"""你是加密货币交易团队的首席交易员，负责分析市场数据并制定交易计划草案。你没有裁决权，但必须给出最完整、最清晰的推演供审计和委员会复核。

【铁律】
1. 只能使用下方市场数据，严禁任何外部知识或记忆数值。
2. 每一步必须按 [分析数据] → [第一反应] → [自我质疑] → [最终结论] 结构推进，缺一不可。
3. [分析数据] 中必须以填空方式直引表格中的对应数值，不可省略任何一项。
4. 最终方向必须与 direction_bias 同向。若相反，必须在第七步列出至少三项基于具体数据字段的反驳理由，并自动将置信度降为 'low'。
5. 所有步骤（第一到第七步）必须完整写入 "reasoning" 字段，不可为空。若 "reasoning" 字段为空，策略无效。

【{symbol} 市场数据】
现价：{mark_price_str}
价格7日分位：{price_percentile_str}%
ATR(4h)：{atr_str}
1h波动率：{atr_1h_ratio_str}%
波动因子：{vol_factor_str}
CGDI：{cgdi_str}（分位{cgdi_perc_str}%）
恐慌贪婪：{fear_greed}（7日前{fear_greed_prev}）

清算池：
上方{above_liq_str}B（簇{above_cluster}，触发距{above_trigger}点）
下方{below_liq_str}B（簇{below_cluster}，触发距{below_trigger}点）
清算比值：{liq_ratio_str}

大额挂单：
卖单墙{large_sell_str}M美元，买单墙{large_buy_str}M美元
压迫比：{pressure_str}（正值=卖压大，负值=买压大）

订单簿：
买方挂单{ob_bids_str}M / 卖方挂单{ob_asks_str}M
失衡率：{ob_imbalance_str}（正值=买盘强）
诱饵风险系数：{lure_str}（>0.5表示显著诱盘风险）

持仓与情绪：
OI：{oi_str}B（分位{oi_perc_str}%，24h变化{oi_chg_str}%）
全市场聚合OI 24h变化：{agg_oi_chg_str}%
资金费率：{fund_rate_str}%（分位{fund_perc_str}%，动量{fund_mom_str}）

多空结构：
大户持仓多空比：{top_ls_str}（分位{top_ls_perc_str}%）
全市场账户多空比：{global_ls_str}
散户-顶级背离指数：{div_str}（正值=大户看多/散户看空）

爆仓：
1h内多头爆仓{long_liq_str}M美元，空头爆仓{short_liq_str}M美元
爆仓偏空比：{liq_bias_str}（正值=空头爆得多）

资金流：
CVD斜率：{cvd_slope_str}
CVD加速度：{cvd_accel_str}
CVD均值（量级）：{cvd_mean_str}M
主动买卖比(1h)：{taker_str}（>1买方主导）

净流入/流出：
5分钟：{nf5m_str}M
1小时：{nf1h_str}M
24小时：{nf24h_str}M
交易所BTC余额24h变化：{exch_btc_str} BTC

现货资金流：
现货1h净流：{spot_1h_str}M，现货24h净流：{spot_24h_str}M
现货/期货资金流背离度：{spot_div_str}（正值=同向，负值=背离）

期权/汇率：
最大痛点：{max_pain_str}
P/C比：{pc_str}
ETH/BTC汇率：{eth_btc_str}（7日均{eth_btc_ma_str}，7日分位{eth_btc_perc_str}%）

宏观与链上新增数据：
合约基差：{basis_str}（分位{basis_perc_str}%）
稳定币总市值：{stable_mcap_str}B，7日趋势：{stable_trend_str}%
BTC市值占比：{btc_dom_str}%，7日趋势：{btc_dom_trend_str}%
LTH已实现价格：{lth_rp_str}  STH已实现价格：{sth_rp_str}
LTH SOPR：{lth_sopr_str}  STH SOPR：{sth_sopr_str}（>1盈利，<1亏损）
借贷利率：{borrow_str}%

{cross_context}
{constraint_note}

【系统客观锚点】
方向综合评分（direction_bias）：{direction_bias:.3f}
（正值=偏多，负值=偏空，绝对值>0.4为强方向信号；最终方向必须与其同向）

--------------------------------
第一步：环境定调
[分析数据] 直引填空：价格7日分位__%，1h波动率__%，波动因子__，CGDI当前__（分位__%），恐慌贪婪__（7日前__）。合约基差__（分位__%），稳定币市值趋势__%，借贷利率__%。
[第一反应] 当前市场温度与稳定性如何？是否适合狩猎？
[自我质疑]
- CGDI与恐慌贪婪是否同时指向极值？基差是否异常？波动因子是否>2.0？
- 稳定币市值是扩张还是萎缩？借贷利率是否在极端水平？
[最终结论]
环境评分（0-100）：__分
仓位硬上限：≥70分 → 可重仓；40-70 → 中/轻仓；<40 → 等待/无仓位。

第二步：对手盘痛苦度测算
[分析数据] 直引填空：
OI分位__%，OI 24h变化__%，全市场聚合OI 24h变化__%，费率当前__%，费率分位__%，费率动量__，大户多空比分位__%，全市场账户多空比__，散户-顶级背离指数__，1h多头爆仓__M美元，1h空头爆仓__M美元，爆仓偏空比__。链上成本：LTH RP __，STH RP __，STH SOPR __。
[第一反应] 哪一方（多头/空头）正在被市场系统性惩罚？链上数据显示谁在盈亏边缘？
[自我质疑]
1. 用OI变化+全市场OI变化判断资金进出方向。
2. 背离指数与爆仓偏空比串联，判断反向猎杀概率。
3. STH SOPR 与 STH RP 判断短期持有者是否整体亏损。
[最终结论]
多头痛苦度（0-100）：__分；空头痛苦度（0-100）：__分
反向猎杀倾向：向上猎杀（挤空头）/ 向下猎杀（挤多头）/ 暂无
反向猎杀概率：高 / 中 / 低

第三步：流动性地形与猎物定位
[分析数据] 直引填空：上方清算__B（上沿距现价__点，远边界距__点），下方清算__B（下沿距现价__点，远边界距__点），清算比值__:1。大额挂单：卖方__M美元，买方__M美元，压迫比__。订单簿失衡率__，诱饵风险系数__。
[第一反应] 最短触发距在哪个方向？初始猎物在哪侧？
[自我质疑]
1. 计算有效引爆距：上方触发距 + 卖方大单缓冲 ≈ __；下方触发距 + 买方大单缓冲 ≈ __。短者为做市商最可能穿刺方向。
2. 诱饵检测：若清算密集侧与大单压迫方向相同，则可能是诱饵陷阱。
3. 对立池威胁评估：对立池清算比值>1.5意味着反向威胁高。
[最终结论]
第一段最可能运动方向：向上 / 向下
猎物定性：真实猎物 / 诱饵陷阱；真实度：高 / 中 / 低

第四步：资金流多轨验证
[分析数据] 直引填空：CVD斜率__，CVD加速度__，CVD量级（均值）__M，主动买卖比(1h)__，5分钟净流__M，1小时净流__M，24小时净流__M，现货1h净流__M，现货24h净流__M，现货/期货背离度__，交易所BTC余额24h变化__BTC。
[第一反应] 资金流整体方向与第三步的猎物方向是否一致？现货与期货是否共振？
[自我质疑]
1. CVD与加速度：加速/衰竭/稳定？
2. 主动买卖比验证：CVD与主动买卖是否一致？
3. 多周期净流共振：5m/1h/24h是否同向？
4. 现货与期货是否背离？若背离需降级。
[最终结论]
资金流共振评级：正向共振 / 中性 / 严重背离
若背离，则对第三步方向施加降级。

第五步：辅助信号扫描
[分析数据] 直引填空：期权最大痛点__，现价距痛点__点（__%），P/C比__。ETH/BTC汇率：当前__，7日均__，7日分位__%。BTC市值占比__%，占比趋势__%。
[第一反应] 期权痛点磁吸效应如何？P/C比指向极端情绪吗？风险偏好如何？
[自我质疑]
1. 痛距判断：若现价距痛点超过2个ATR，磁吸效应较强。
2. P/C比极值：>1.2极度看跌，<0.7极度看涨，极值常为反向信号。
3. BTC.D趋势：上升=避险，下降=风险偏好升。与ETH/BTC结合判断。
[最终结论]
辅助信号对主逻辑（第三步+第四步方向）：支持 / 中性 / 警告

第六步：跨币种验证
[分析数据] 直引填空（必须写数值）：
① 清算方向：{symbol}清池偏向[上方/下方]，{cross_symbol}清池偏向[上方/下方] → 一致/矛盾？
② CVD方向：{symbol}CVD[正/负]，{cross_symbol}CVD[正/负] → 共振/背离？
③ 顶多空分位：{symbol}[__]%，{cross_symbol}[__]% → 谁更极端？
④ OI 24h变化：{symbol}[__]%，{cross_symbol}[__]% → 资金在流入哪个币种？
⑤ 爆仓偏空比：{symbol}[__]，{cross_symbol}[__] → 哪边压力更大？
[第一反应] 两个币种整体方向一致还是分裂？
[自我质疑]
若出现重大分歧，是结构性轮动（单币种策略仍有效）还是系统性风险（必须降级）？
跨币种一致性评分：每项一致+1，矛盾-1。五项合计范围-5~+5。
[最终结论]
跨币种一致性评分：__分
对主逻辑的影响：维持 / 降级置信度 / 强制转为中性
新增风险点：__

第七步：制定交易计划
[综合前六步结论，并结合系统客观锚点 **direction_bias = {direction_bias:.3f}** 给出最终策略，最终方向必须与 direction_bias 符号一致。若不一致，必须在下方逐条列出至少三项基于具体数据字段的反驳理由，并自动将置信度降为 'low'。若输出 neutral，所有价格字段为0，仓位为none。]
[价格路径推演]：综合“流动性猎杀、对手盘心理、博弈论”，用1-2句话推演该币最可能的价格走势。
[最终合约策略]（不能省略）：
- 币种：{symbol}
- 方向：[做多/做空/观望]
- 现价：
- 仓位：[轻仓/中仓/重仓/无]
- 置信度：[高/中/低]
- 入场区间：[__-__] (依据：__)
- 止损：[__] (依据：__)
- 止盈：[__] (依据：__)
- 说明：[一句话指令或观望触发条件]

【强制输出要求】
请将以上七步完整推演全部写入 "reasoning" 字段，不可省略任何一步。若 "reasoning" 为空或内容不完整，该策略将被系统自动驳回。
输出JSON（不要加代码块标记）：
{{
  "direction": "做多 / 做空 / 观望",
  "confidence": "高 / 中 / 低",
  "position_size": "重仓 / 中仓 / 轻仓 / 无",
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "execution_plan": "一句话指令（若观望则写触发条件）",
  "reasoning": "务必包含第一步到第七步的完整推演，每步均需有数值和分析、第一反应、自我质疑、最终结论，不得空缺。",
  "risk_note": "核心风险（20字内）",
  "final_strategy": "请将第七步最终合约策略填入此字段，格式：\\n- 币种：{symbol}\\n- 方向：[做多/做空/观望]\\n- 现价：\\n- 仓位：[重仓/中仓/轻仓/无]\\n- 入场区间：[__-__] (依据：__)\\n- 止损：[__] (依据：__)\\n- 止盈：[__] (依据：__)\\n- 说明：[__]\\n主动证伪：[__]\\n微观确认：[__]"
}}
"""
    return prompt


# ------------------- 首席交易员调用 -------------------
def call_deepseek(prompt: str, max_retries: int = MAX_RETRIES) -> dict:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=TIMEOUT_SECONDS
    )
    for attempt in range(max_retries):
        try:
            logger.info(f"首席交易员 调用 (尝试 {attempt+1}/{max_retries}) [模型: {FAST_MODEL}]")
            resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                timeout=TIMEOUT_SECONDS
            )
            logger.info(f"实际调用的模型: {resp.model}")
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
            raw_dir = s.get("direction", "")
            if raw_dir in dir_map:
                s["direction"] = dir_map[raw_dir]
            else:
                reasoning_text = s.get("reasoning", "")
                final_decision_match = re.search(r'方向[：:]\s*(做多|做空|观望|long|short|neutral)', reasoning_text)
                if final_decision_match:
                    s["direction"] = dir_map.get(final_decision_match.group(1), "neutral")
                else:
                    s["direction"] = "neutral"

            # 仓位标准化
            pos_map = {"轻仓": "light", "中仓": "medium", "重仓": "heavy", "无": "none",
                       "light": "light", "medium": "medium", "heavy": "heavy", "none": "none"}
            raw_pos = s.get("position_size", "")
            s["position_size"] = pos_map.get(raw_pos, "none")

            # 置信度标准化
            conf_map = {"高": "high", "中": "medium", "低": "low",
                        "high": "high", "medium": "medium", "low": "low"}
            raw_conf = s.get("confidence", "")
            s["confidence"] = conf_map.get(raw_conf, "medium")

            s.setdefault("execution_plan", "")
            s.setdefault("reasoning", "")
            s.setdefault("risk_note", "")
            s["_model_used"] = resp.model
            return s

        except Exception as e:
            logger.warning(f"首席交易员调用失败: {e}")
            if attempt < max_retries - 1:
                wait_time = RETRY_BASE_WAIT ** (attempt + 1)
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                return {
                    "direction": "neutral",
                    "confidence": "low",
                    "position_size": "none",
                    "entry_price_low": 0,
                    "entry_price_high": 0,
                    "stop_loss": 0,
                    "take_profit": 0,
                    "execution_plan": "模型调用失败，人工介入",
                    "reasoning": "调用失败",
                    "risk_note": "模型调用失败",
                    "final_strategy": "调用失败",
                    "_model_used": "fallback"
                }


# ------------------- 风控审计官 -------------------
def build_reviewer_prompt(original_strategy: dict, data: dict, symbol: str) -> str:
    direction_bias = data.get('direction_bias', 0.0)
    liq_bias = data.get('liquidity_bias', 'neutral')
    bias_map = {'long': '偏向上方', 'short': '偏向下止', 'neutral': '无偏向'}

    return f"""你是我们加密货币团队的风控审计官。请基于【市场数据】逐项核查首席交易员的策略，找出可能存在的遗漏、矛盾或与数据不符之处，按以下模板输出审计报告。

【交易标的】{symbol}
【系统锚点】direction_bias={direction_bias:.3f}（>0偏多，<0偏空，|>0.4|强方向）；liquidity_bias={bias_map.get(liq_bias)}

【交易员推演】
{original_strategy.get('reasoning', '无')}

【交易员策略】
方向：{original_strategy.get('direction')}　置信度：{original_strategy.get('confidence')}　仓位：{original_strategy.get('position_size')}
入场：{original_strategy.get('entry_price_low')} - {original_strategy.get('entry_price_high')}
止损：{original_strategy.get('stop_loss')}　止盈：{original_strategy.get('take_profit')}

【审计规则】
- 使用以下五节标题，每节内部用“- ”开头逐条列出发现。
- 每条发现末尾必须标注 `[严重性：高]`、`[严重性：中]` 或 `[严重性：低]`。
- 某节若确实没有问题，只需写入无问题的声明（如“已覆盖所有应分析的关键指标”）。
- 所有分析必须严格参照提供的数据，不可主观臆测。
- 只输出报告本身，不要添加额外解释。
---
【审计报告】
一、遗漏指标与分析缺失
- [若有遗漏，按此格式：在[步骤X/决策点]中，交易员未分析已提供的[指标名称/数据项]。该指标显示[具体数值/信号]，若纳入分析将[强化/削弱/推翻]当前方向判断。 [严重性：高/中/低]]
- 已覆盖所有应分析的关键指标。（若无遗漏）
二、数据与解读错误
- 在[步骤X]中，交易员声称[数值/解读]，但实际数据为[数值/正确含义]。此错误[是否影响方向判断]。 [严重性：高/中/低]
- 未发现数据或解读错误。（若无错误）
三、逻辑错误
- [错误类型]在[步骤X]：[描述]。 [严重性：高/中/低]
- 未发现明显逻辑错误。（若无错误）
四、关键反证提示
- 在[步骤X]中，策略依据[数据A]得出[结论]，但已提供的[数据B]显示[相反信号]，二者构成矛盾，未被交易员处理。 [严重性：高/中/低]
- 未发现关键反证被忽略。（若无矛盾）
五、博弈层面审视
- 基于清算池数据，上方/下方清算密集区的吸引力更[强/弱]，做市商猎杀方向倾向于[上方/下方/均衡]。当前策略止损位[在/不在]该路径上，[若在，描述风险]。 [严重性：高/中/低]
- 预设入场区间与[大额挂单墙/清算密集区]重合，可能成为对手盘流动性来源，[具体距离/重合度]。 [严重性：高/中/低]
- 若数据不足以判断猎杀方向或结构位重合，请写“数据不足，无法判定”，但仍需保持条目格式。
"""


def call_reviewer(original_strategy: dict, data: dict, symbol: str) -> dict:
    prompt = build_reviewer_prompt(original_strategy, data, symbol)
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=TIMEOUT_SECONDS
    )
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"风控审计官 调用 (尝试 {attempt+1}/{MAX_RETRIES}) [模型: {FAST_MODEL}]")
            resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                timeout=120
            )
            logger.info(f"实际调用的模型: {resp.model}")
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip():
                raise ValueError("审计官响应为空")

            severity_counts = {"高": 0, "中": 0, "低": 0}
            for line in content.split('\n'):
                for level in ["高", "中", "低"]:
                    if f"严重性：{level}" in line:
                        severity_counts[level] += 1

            verdict = "通过"
            if severity_counts["高"] > 0:
                verdict = "驳回"
            elif severity_counts["中"] > 0 or severity_counts["低"] > 0:
                verdict = "存疑"

            return {
                "verdict": verdict,
                "full_report": content,
                "severity_counts": severity_counts,
                "_model_used": resp.model
            }
        except Exception as e:
            logger.warning(f"风控审计官调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                logger.warning("风控审计官所有重试均失败，标记为通过，跳过审计")
                return {"verdict": "通过", "full_report": "审计官调用失败，跳过审计", "_model_used": "fallback"}


# ------------------- 交易委员会 -------------------
def build_judge_prompt(original_strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> str:
    direction_bias = data.get('direction_bias', 0.0)
    mark_price = data.get('mark_price', 0.0)
    atr_15m = data.get('atr_15m', 0.0)
    above_trigger = data.get('above_trigger', 'N/A')
    below_trigger = data.get('below_trigger', 'N/A')
    large_order_pressure = data.get('large_order_pressure', 0.0)

    orig_dir = original_strategy.get('direction', 'neutral')
    entry_l = original_strategy.get('entry_price_low', 0)
    entry_h = original_strategy.get('entry_price_high', 0)
    stop_l = original_strategy.get('stop_loss', 0)
    tp_l = original_strategy.get('take_profit', 0)

    report = reviewer_report.get('full_report', '无审计报告')

    return f"""你是最终决策的**独立交易委员会主席**，拥有二十年加密货币短线合约交易经验。
你的职责是：基于【市场数据】，公正审核【首席交易员的策略】和【风控审计报告】，给出最合理的执行方案。

【标的信息】{symbol}　现价：{mark_price:.2f}　ATR(15m)：{atr_15m:.2f}

【核心锚点】
direction_bias = {direction_bias:.3f}（>0偏多，<0偏空，|>0.4|为强方向信号）
清算触发距：上方{above_trigger}点 / 下方{below_trigger}点
大单压迫比：{large_order_pressure:.3f}（正值=卖压强）

【交易员策略】
方向：{orig_dir}　置信度：{original_strategy.get('confidence', 'N/A')}　仓位：{original_strategy.get('position_size', 'N/A')}
入场：{entry_l} - {entry_h}
止损：{stop_l}
止盈：{tp_l}
推演：{original_strategy.get('reasoning', '无')}

【审计报告】
{report}

【裁决规则】
1. 事实为上：所有判断必须引用具体数据字段和数值，不得凭感觉或记忆。
2. 独立公正：不听信交易员的一面之词，也不盲从审计官的指控。必须亲自核验每项审计指控。
3. 逻辑自洽：最终结论（方向、仓位、价格）必须与核验依据形成闭环，不能自相矛盾。
4. 锚点优先：若|direction_bias|>0.4且交易方向相反且无充分数据证伪，必须推翻原策略。
5. 参数验证：入场、止损、止盈必须基于ATR、清算池边界或大单挂单价格。

【输出格式】

📋 裁决说明：
  *方向锚点检查：direction_bias={direction_bias:.3f}，交易方向={orig_dir}，一致/矛盾。
   （若矛盾，说明交易员是否列出三项以上证伪数据及是否成立）

  *审计指控裁决（逐条）：
   1. 指控内容：[原文概括]
      裁决结论：采纳/驳回/部分采纳
      核验依据：（引用字段+数值）
      a) 核对数值是否一致。
      b) 若数值不一致 → 直接驳回，说明审计官的错误。
      c) 若数值一致 → 判断是否实质影响交易方向。若不影响方向，可标记为“部分采纳”。
      反证风险评估（高/中/低）：提出至少一条可能挑战本裁决结论的证据，包含该证据是什么、为何构成威胁、以及为何最终排除。若确实无矛盾可写“无”，但必须简述搜寻范围。
   2. 指控内容：[...]
      ...

  *主动补充：市场数据中是否存在审计官遗漏但可能影响方向的反向信号？若有，必须指出并同样进行反证风险评估。

📌 最终判决：[维持原判 / 推翻]（二选一，不要输出其它字符）
  *核心逻辑：说明最终判决的依据，必须提供站的住脚的依据或推论。
  若推翻原方向，必须满足：
  1) 明确指出原方向依赖的一个或多个核心假设，并用数据字段证伪。
  2) 给出至少两个支持新方向的独立数据字段（字段名+数值）。
  3) 新方向必须与 direction_bias 在逻辑上自洽，若矛盾需给出强证据链解释。

🎯 合约策略（以下数据缺一不可）：
  *币种：{symbol}
  *方向：[做多/做空/观望]
  *现价：
  *仓位：[重仓/中仓/轻仓/无]
  *入场区间：[价格下限-价格上限]（依据）
  *止损：[价格]（依据）
  *止盈：[价格]（依据）
  *说明：[一句话指令或观望触发条件]
⚠️ 风险说明：[一句话]
"""


def call_judge(original_strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    prompt = build_judge_prompt(original_strategy, reviewer_report, data, symbol)
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=TIMEOUT_SECONDS
    )
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"交易委员会 调用 (尝试 {attempt+1}/{MAX_RETRIES}) [模型: {REASONING_MODEL}]")
            resp = client.chat.completions.create(
                model=REASONING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                timeout=120
            )
            logger.info(f"实际调用的模型: {resp.model}")
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip():
                raise ValueError("交易委员会响应为空")

            # 提取最终判决
            verdict = "维持原判"
            verdict_match = re.search(r'📌\s*最终判决[：:]\s*(.*)', content)
            if verdict_match:
                verdict_line = verdict_match.group(1).strip()
                clean_line = verdict_line.replace('*', '').replace('`', '').strip()
                if "维持" in clean_line:
                    verdict = "维持原判"
                else:
                    verdict = "推翻"
            else:
                clean_tail = content[-500:].replace('*', '').replace('`', '')
                if "推翻" in clean_tail and "维持" not in clean_tail:
                    verdict = "推翻"
                logger.warning("未找到 📌 最终判决 行，从全文推断")

            # 提取风险说明
            risk_text = ""
            risk_match = re.search(r'⚠️\s*风险说明[：:]\s*(.*)', content, re.DOTALL)
            if risk_match:
                risk_text = risk_match.group(1).strip().split('\n')[0]

            # 维持原判
            if verdict == "维持原判":
                judge_result = {
                    "judge_C": {
                        "final_verdict": "维持原判",
                        "final_direction": original_strategy.get("direction", "neutral"),
                        "final_confidence": original_strategy.get("confidence", "medium"),
                        "final_position_size": original_strategy.get("position_size", "none"),
                        "entry_price_low": original_strategy.get("entry_price_low", 0) or 0,
                        "entry_price_high": original_strategy.get("entry_price_high", 0) or 0,
                        "stop_loss": original_strategy.get("stop_loss", 0) or 0,
                        "take_profit": original_strategy.get("take_profit", 0) or 0,
                        "execution_plan": original_strategy.get("execution_plan", ""),
                        "reasoning": content,
                        "risk_note": risk_text if risk_text else original_strategy.get('risk_note', ''),
                        "title_line": "📌 最终判决: 维持原判",
                        "current_price": data.get("mark_price", 0),
                        "_model_used": resp.model
                    }
                }
                logger.info(f"交易委员会裁决: 维持原判 (沿用原策略), 模型: {resp.model}")
                return judge_result

            # 推翻，解析执行指令
            direction = "neutral"
            position_size = "none"
            entry_low = entry_high = stop_loss = take_profit = 0.0
            execution_plan = ""
            current_price = data.get("mark_price", 0)

            exec_match = re.search(r'🎯\s*(?:合约策略|执行指令)[：:]\s*(.*?)(?=📋|📌|⚠️|$)', content, re.DOTALL)
            exec_block = ""
            if exec_match:
                exec_block = exec_match.group(1).strip()
                risk_split = re.split(r'⚠️\s*风险说明[：:]', exec_block, maxsplit=1)
                if len(risk_split) > 1:
                    exec_block = risk_split[0].strip()
                    if not risk_text:
                        risk_text = risk_split[1].strip().split('\n')[0]

            if exec_block:
                clean_block = re.sub(r'^\s*\*\s*', '', exec_block, flags=re.MULTILINE)

                dir_match = re.search(r'方向[：:]\s*([^\s(（*\n]+)', clean_block)
                if dir_match:
                    raw_dir = dir_match.group(1).replace('*', '').strip()
                    dir_map = {"做多": "long", "做空": "short", "观望": "neutral",
                               "long": "long", "short": "short", "neutral": "neutral"}
                    direction = dir_map.get(raw_dir, "neutral")
                else:
                    first_200 = exec_block[:200].replace('*', '')
                    if "做多" in first_200: direction = "long"
                    elif "做空" in first_200: direction = "short"
                    elif "观望" in first_200: direction = "neutral"

                pos_match = re.search(r'仓位[：:]\s*([^\s(（*\n]+)', clean_block)
                if pos_match:
                    raw_pos = pos_match.group(1).replace('*', '').strip()
                    pos_map = {"轻仓": "light", "中仓": "medium", "重仓": "heavy", "无": "none", "无仓位": "none"}
                    position_size = pos_map.get(raw_pos, "none")

                price_match = re.search(r'现价[：:]\s*([\d.]+)', clean_block)
                if price_match:
                    current_price = float(price_match.group(1))

                entry_match = re.search(r'入场区间[：:]\s*([\d.]+)\s*[-–]\s*([\d.]+)', clean_block)
                if entry_match:
                    entry_low = float(entry_match.group(1))
                    entry_high = float(entry_match.group(2))

                stop_match = re.search(r'止损[：:]\s*([\d.]+)', clean_block)
                if stop_match:
                    stop_loss = float(stop_match.group(1))

                tp_match = re.search(r'止盈[：:]\s*([\d.]+)', clean_block)
                if tp_match:
                    take_profit = float(tp_match.group(1))

                plan_match = re.search(r'说明[：:]\s*(.*)', clean_block)
                if plan_match:
                    execution_plan = plan_match.group(1).strip()

            if direction == "neutral":
                entry_low = entry_high = stop_loss = take_profit = 0.0
                position_size = "none"
                if not execution_plan:
                    execution_plan = "观望"

            judge_result = {
                "judge_C": {
                    "final_verdict": "推翻",
                    "final_direction": direction,
                    "final_confidence": "medium",
                    "final_position_size": position_size,
                    "entry_price_low": entry_low,
                    "entry_price_high": entry_high,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "execution_plan": execution_plan,
                    "reasoning": content,
                    "risk_note": risk_text if risk_text else "",
                    "title_line": "📌 最终判决: 推翻",
                    "current_price": current_price,
                    "_model_used": resp.model
                }
            }
            logger.info(f"交易委员会裁决: 推翻, 方向: {direction}, 模型: {resp.model}")
            return judge_result

        except Exception as e:
            logger.warning(f"交易委员会调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                return {"judge_C": {"final_verdict": "维持原判", "verdict_level": "A", "_model_used": "fallback"}}


def apply_final_verdict(original_strategy: dict, judge_result: dict, reviewer_report: dict = None) -> dict:
    verdict = judge_result.get("judge_C", {}).get("final_verdict", "维持原判")
    final = judge_result.get("judge_C", {})
    logger.info(f"应用最终决议: {verdict}")

    original_strategy["_reviewed"] = True
    original_strategy["_original_direction"] = original_strategy.get("direction")
    original_strategy["_review_verdict"] = verdict
    original_strategy["_judge_data"] = final
    original_strategy["_model_used"] = final.get("_model_used", "")

    if verdict == "维持原判":
        if "risk_note" in final:
            original_strategy["risk_note"] = final["risk_note"]
        original_strategy["_judge_reasoning"] = final.get("reasoning", "")
    else:
        new_dir = final.get("final_direction", "neutral")
        if new_dir == "neutral":
            _force_neutral(original_strategy, "交易委员会推翻并改为观望")
        else:
            original_strategy["direction"] = new_dir
            original_strategy["confidence"] = final.get("final_confidence", "medium")
            original_strategy["position_size"] = final.get("final_position_size", "none")
            original_strategy["entry_price_low"] = final.get("entry_price_low", 0) or 0
            original_strategy["entry_price_high"] = final.get("entry_price_high", 0) or 0
            original_strategy["stop_loss"] = final.get("stop_loss", 0) or 0
            original_strategy["take_profit"] = final.get("take_profit", 0) or 0
            original_strategy["execution_plan"] = final.get("execution_plan", "")
            original_strategy["risk_note"] = final.get("risk_note", "")
        original_strategy["_judge_reasoning"] = final.get("reasoning", "")

    if original_strategy.get("direction") == "neutral":
        original_strategy["entry_price_low"] = 0
        original_strategy["entry_price_high"] = 0
        original_strategy["stop_loss"] = 0
        original_strategy["take_profit"] = 0
        original_strategy["position_size"] = "none"
        if not original_strategy.get("execution_plan"):
            original_strategy["execution_plan"] = "观望"

    return original_strategy


# ====================================================
# 需要同步修改 CoinGlassClient._build_main_data 的代码片段
# (请复制到 CoinGlassClient 类中的 _build_main_data 方法对应位置)
# ====================================================
"""
1) 稳定币市值（替换原 stablecoin_mcap_data 处理块）
if stablecoin_mcap_data and isinstance(stablecoin_mcap_data, list) and len(stablecoin_mcap_data) > 0:
    first_item = stablecoin_mcap_data[0]
    data_list = first_item.get("data_list", [])
    if data_list:
        stablecoin_mcap_current = float(data_list[-1])
        if len(data_list) >= 7:
            stablecoin_trend = (data_list[-1] - data_list[-7]) / (data_list[-7] + 1e-8) * 100

2) 比特币占比（替换原 btc_dom_data 处理块）
if btc_dom_data and len(btc_dom_data) > 0:
    dom_values = [float(d.get("bitcoin_dominance", 0)) for d in btc_dom_data]
    btc_dom_current = dom_values[-1]
    if len(dom_values) >= 7:
        btc_dom_trend = (dom_values[-1] - dom_values[-7]) / (dom_values[-7] + 1e-8) * 100

3) 借贷利率（替换原 borrow_rate_data 处理块）
if borrow_rate_data and len(borrow_rate_data) > 0:
    rates = [float(d.get("interest_rate", 0)) for d in borrow_rate_data]
    borrow_rate_current = rates[-1]
"""
