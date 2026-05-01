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
FAST_MODEL = "deepseek-v4-flash"          # 交易员 & 审计官
REASONING_MODEL = "deepseek-v4-pro"       # 交易委员会 (深度思考)

# ------------------- 首席交易员：生成最初计划 -------------------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    if cross_symbol is None:
        if symbol == "BTC": cross_symbol = "ETH"
        elif symbol == "ETH": cross_symbol = "BTC"
        else: cross_symbol = "ETH"

    # 跨币种文本
    cross_context = ""
    if eth_data:
        # 始终构造跨币种数据展示（无论是否完整），缺失字段默认值仍会显示
        cross_context = f"""
【{cross_symbol} 跨币种数据 - 仅用于第六步】
现价：{eth_data.get('mark_price', 0):.2f}
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

    # 缺失约束
    data_quality = data.get("data_quality", {})
    core_fields = ["atr_15m", "above_liq", "below_liq", "cvd_slope", "global_ls_ratio", "taker_ratio_1h", "large_orders", "liq_history"]
    missing_core = [k for k in core_fields if data_quality.get(k) == "❌ 缺失"]
    constraint_note = ""
    if missing_core:
        constraint_note = f"\n【重要约束】以下核心数据缺失：{', '.join(missing_core)}。你必须将最终置信度设为 'low'；若清算数据缺失，则必须输出 'neutral'。\n"

    prompt = f"""你是加密货币交易团队的首席交易员，负责分析市场数据并制定交易计划草案。你没有裁决权，但必须给出最完整、最清晰的推演供审计和委员会复核。

【铁律】
1. 只能使用下方市场数据，严禁任何外部知识或记忆数值。
2. 每一步必须按 [分析数据] → [第一反应] → [自我质疑] → [最终结论] 结构推进，缺一不可。
3. [分析数据] 中必须以填空方式直引表格中的对应数值，不可省略任何一项。
4. 最终方向必须与 direction_bias 同向。若相反，必须在第七步列出至少三项基于具体数据字段的反驳理由，并自动将置信度降为 'low'。

【{symbol} 市场数据】
现价：{data['mark_price']:.2f}
价格7日分位：{data['price_percentile']:.0f}%
ATR(4h)：{data['atr']:.2f}
1h波动率：{data['atr_1h_ratio']:.1f}%
波动因子：{data['vol_factor']:.2f}
CGDI：{data['cgdi_current']:.0f}（分位{data['cgdi_percentile']:.0f}%）
恐慌贪婪：{data['fear_greed']}（7日前{data['fear_greed_prev_7d']}）

清算池：
上方{data['above_liq']/1e9:.2f}B（簇{data['above_cluster']}，触发距{data['above_trigger']}点）
下方{data['below_liq']/1e9:.2f}B（簇{data['below_cluster']}，触发距{data['below_trigger']}点）
清算比值：{data['liq_ratio']:.2f}

大额挂单：
卖单墙{data['large_sell_value']/1e6:.1f}M美元，买单墙{data['large_buy_value']/1e6:.1f}M美元
压迫比：{data['large_order_pressure']:.3f}（正值=卖压大，负值=买压大）

订单簿：
买方挂单{data['orderbook_bids']/1e6:.1f}M / 卖方挂单{data['orderbook_asks']/1e6:.1f}M
失衡率：{data['orderbook_imbalance']:.3f}（正值=买盘强）
诱饵风险系数：{data['lure_risk_factor']:.2f}（>0.5表示显著诱盘风险）

持仓与情绪：
OI：{data['oi']/1e9:.2f}B（分位{data['oi_percentile']:.0f}%，24h变化{data['oi_change_24h']:+.1f}%）
全市场聚合OI 24h变化：{data['agg_oi_change_24h']:+.1f}%
资金费率：{data['funding_rate']:.4f}%（分位{data['funding_percentile']:.0f}%，动量{data['funding_momentum']:.6f}）

多空结构：
大户持仓多空比：{data['top_ls_ratio']:.2f}（分位{data['top_ls_percentile']:.0f}%）
全市场账户多空比：{data['global_ls_ratio']:.2f}
散户-顶级背离指数：{data['retail_whale_divergence']:.3f}（正值=大户看多/散户看空，向上挤压倾向）

爆仓：
1h内多头爆仓{data['long_liq_1h']/1e6:.1f}M美元，空头爆仓{data['short_liq_1h']/1e6:.1f}M美元
爆仓偏空比：{data['liq_bias_1h']:.3f}（正值=空头爆得多，偏空信号）

资金流：
CVD斜率：{data['cvd_slope']:.4f}
CVD加速度：{data['cvd_acceleration']:.4f}
CVD均值（量级）：{data['cvd_mean']:.2f}M
主动买卖比(1h)：{data['taker_ratio_1h']:.3f}（>1买方主导，<1卖方主导）

净流入/流出：
5分钟：{data['netflow_5m']/1e6:.1f}M
1小时：{data['netflow_1h']/1e6:.1f}M
24小时：{data['netflow_24h']/1e6:.1f}M
交易所BTC余额24h变化：{data['exchange_btc_change_24h']:+.0f} BTC

期权/汇率：
最大痛点：{data['max_pain']:.2f}
P/C比：{data['put_call_ratio']:.4f}
ETH/BTC汇率：{data['eth_btc_ratio']:.4f}（7日均{data['eth_btc_ma_7d']:.4f}，7日分位{data['eth_btc_percentile']:.0f}%）

{cross_context}
{constraint_note}

【系统客观锚点】
方向综合评分（direction_bias）：{data['direction_bias']:.3f}
（正值=偏多，负值=偏空，绝对值>0.4为强方向信号；最终方向必须与其同向）

--------------------------------
第一步：环境定调
[分析数据] 直引填空：价格7日分位__%，1h波动率__%，波动因子__，CGDI当前__（分位__%），恐慌贪婪__（7日前__）。
[第一反应] 当前市场温度与稳定性如何？是否适合狩猎？
[自我质疑]
- CGDI与恐慌贪婪是否同时指向极值区域？若CGDI分位>85且恐慌贪婪>75，市场过热，需警惕反转。
- 波动因子是否超过2.0？若是，市场处于异常高波动，即使方向明确也可能被剧烈震荡洗出。
- 波动是恐慌驱动还是贪婪驱动？结合价格分位判断。
[最终结论]
环境评分（0-100）：__分
仓位硬上限：≥70分 → 可重仓；40-70 → 中/轻仓；<40 → 等待/无仓位。

第二步：对手盘痛苦度测算
[分析数据] 直引填空：
OI分位__%，OI 24h变化__%，全市场聚合OI 24h变化__%，费率当前__%，费率分位__%，费率动量__，大户多空比分位__%，全市场账户多空比__，散户-顶级背离指数__，1h多头爆仓__M美元，1h空头爆仓__M美元，爆仓偏空比__。
[第一反应] 哪一方（多头/空头）正在被市场系统性惩罚？
[自我质疑]
1. 用OI变化+全市场OI变化判断：市场整体是在加仓还是撤离？加仓方向是谁？
2. 用背离指数+爆仓偏空比串联：若背离>0.5且爆仓偏空比<-0.3 → 向上猎杀概率极高。反之背离<-0.5且偏空比>0.3 → 向下猎杀概率高。
3. 用费率分位+费率动量判断：极度拥挤的一方是否已经开始松动？
[最终结论]
多头痛苦度（0-100）：__分；空头痛苦度（0-100）：__分
反向猎杀倾向：向上猎杀（挤空头）/ 向下猎杀（挤多头）/ 暂无
反向猎杀概率：高 / 中 / 低

第三步：流动性地形与猎物定位
[分析数据] 直引填空：上方清算__B（上沿距现价__点，远边界距__点），下方清算__B（下沿距现价__点，远边界距__点），清算比值__:1。
大额挂单：卖方__M美元，买方__M美元，压迫比__。
订单簿失衡率__，诱饵风险系数__。
[第一反应] 最短触发距在哪个方向？初始猎物在哪侧？
[自我质疑]
1. 计算有效引爆距：有效上方引爆距 = 上方触发距 + 卖方大单缓冲（卖方大单量/均量*ATR）≈ __；有效下方引爆距 = 下方触发距 + 买方大单缓冲 ≈ __。短者为做市商最可能穿刺方向。
2. 诱饵检测：若清算密集侧与大单压迫方向相同，则清算密集很可能是诱饵陷阱。
3. 对立池威胁评估：对立池清算比值>1.5意味着反向威胁高。
[最终结论]
第一段最可能运动方向：向上 / 向下
猎物定性：真实猎物 / 诱饵陷阱；真实度：高 / 中 / 低

第四步：资金流多轨验证
[分析数据] 直引填空：CVD斜率__，CVD加速度__，CVD量级（均值）__M，主动买卖比(1h)__，5分钟净流__M，1小时净流__M，24小时净流__M，交易所BTC余额24h变化__BTC。
[第一反应] 资金流整体方向与第三步的猎物方向是否一致？
[自我质疑]
1. CVD方向与加速度：当前属于加速/衰竭/稳定？
2. CVD量级评估：若量级极小(<0.5M)，信号可能是噪声。
3. 主动买卖比验证：若CVD上升但主动买比<1，说明CVD由被动成交堆积。
4. 多周期净流共振：若5m/1h/24h三级方向一致，信号可靠度高；若5m与24h相反，短期可能有反向波动。
5. 交易所BTC余额：减少=提币持有(偏多)；增加=转入待售(偏空)。
[最终结论]
资金流共振评级：正向共振 / 中性 / 严重背离
若背离，则对第三步方向施加降级。

第五步：辅助信号扫描
[分析数据] 直引填空：期权最大痛点__，现价距痛点__点（__%），P/C比__。
ETH/BTC汇率：当前__，7日均__，7日分位__%。
[第一反应] 期权痛点是否对现价有磁吸效应？P/C比是否指向极端情绪？ETH/BTC反映的风险偏好如何？
[自我质疑]
1. 痛距判断：若现价距痛点超过2个ATR，磁吸效应较强。
2. P/C比极值：>1.2极度看跌，<0.7极度看涨，极值常为反向信号。
3. ETH/BTC分位：高分位=风险偏好高；低分位=避险情绪浓。
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
[综合前六步结论，并结合系统客观锚点 **direction_bias = {data['direction_bias']:.3f}** 给出最终策略，最终方向必须与 direction_bias 符号一致。若不一致，必须在下方逐条列出至少三项基于具体数据字段的反驳理由，并自动将置信度降为 'low'。若输出 neutral，所有价格字段为0，仓位为none。]
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
  "reasoning": "第一步到第七步的完整推演文本，包含所有数据分析、第一反应、自我质疑、最终结论、价格路径推演'。",
  "risk_note": "核心风险（20字内）",
  "final_strategy": "请将第七步最终合约策略填入此字段，格式：\\n- 币种：{symbol}\\n- 方向：[做多/做空/观望]\\n- 现价：\\n- 仓位：[重仓/中仓/轻仓/无]\\n- 入场区间：[__-__] (依据：__)\\n- 止损：[__] (依据：__)\\n- 止盈：[__] (依据：__)\\n- 说明：[__]\\n主动证伪：[__]\\n微观确认：[__]"
}}
"""
    return prompt


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
            s["_model_used"] = resp.model  # 记录实际使用的模型
            return s

        except Exception as e:
            logger.warning(f"首席交易员调用失败: {e}")
            if attempt < max_retries - 1:
                wait_time = RETRY_BASE_WAIT ** (attempt + 1)
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise
    return {}


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
    s["reasoning"] = (s.get("reasoning", "") + f"\n\n[原始信号因校验规则被强制改为观望，原因：{reason}]").strip()
    s["risk_note"] = f"观望。{reason}"


def validate_strategy(s: dict, data: dict = None) -> tuple[bool, str]:
    direction = s.get("direction")
    if direction not in ["long", "short", "neutral"]:
        return False, f"无效方向: {direction}"

    if data:
        atr_15m = data.get("atr_15m", 0)
        above_liq = data.get("above_liq", 0)
        below_liq = data.get("below_liq", 0)
        cvd_slope = data.get("cvd_slope", None)
        direction_bias = data.get("direction_bias", 0.0)

        above_ok = above_liq is not None and above_liq > 0
        below_ok = below_liq is not None and below_liq > 0
        if not above_ok and not below_ok and direction != "neutral":
            _force_neutral(s, "清算数据缺失，强制输出 neutral")
            return True, "已自动修正为观望"

        if atr_15m is not None and atr_15m <= 0 and s.get("confidence") == "high":
            s["confidence"] = "medium"
            logger.warning("核心数据缺失(atr_15m)，置信度强制降级为 medium")
        if cvd_slope is None and s.get("confidence") == "high":
            s["confidence"] = "medium"
            logger.warning("核心数据缺失(cvd_slope)，置信度强制降级为 medium")

        # 方向锚点检查：若|direction_bias|>0.4 且方向相反且置信度高，降级
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


# ------------------- 风控审计官：输出审计报告 -------------------
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
- 预设入场区间与[前高/前低/成交密集区]重合，可能成为对手盘流动性来源，[具体距离/重合度]。 [严重性：高/中/低]
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


# ------------------- 交易委员会：输出最终策略 -------------------
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

🎯 合约策略：
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

            # ========== 1. 从 📌 最终判决 行提取判决（维持原判 / 推翻） ==========
            verdict = "维持原判"          # 默认
            verdict_match = re.search(r'📌\s*最终判决[：:]\s*([^\n]{,30})', content)
            if verdict_match:
                verdict_line = verdict_match.group(1).strip()
                # 只看冒号后紧跟的关键词，忽略括号内的说明
                # 例如 "维持原判（不推翻原策略）" → 含“维持” → 维持原判
                if "维持" in verdict_line:
                    verdict = "维持原判"
                else:
                    verdict = "推翻"
            else:
                logger.warning("未找到 📌 最终判决 行，默认维持原判")

            # ========== 2. 如果是维持原判，完全沿用原策略，不再解析执行指令 ==========
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
                        "risk_note": original_strategy.get("risk_note", ""),
                        "title_line": "📌 最终判决: 维持原判",
                        "current_price": data.get("mark_price", 0),
                        "_model_used": resp.model
                    }
                }
                logger.info(f"交易委员会裁决: 维持原判 (沿用原策略), 模型: {resp.model}")
                return judge_result

            # ========== 3. 推翻时，解析 🎯 合约策略/执行指令 块 提取新参数 ==========
            # 默认值
            direction = "neutral"
            position_size = "none"
            entry_low = entry_high = stop_loss = take_profit = 0.0
            execution_plan = ""
            risk_text = ""
            current_price = data.get("mark_price", 0)

            # 提取执行指令块（🎯 合约策略 / 🎯 执行指令）
            exec_match = re.search(r'🎯\s*(?:合约策略|执行指令)[：:]\s*(.*)', content, re.DOTALL)
            exec_block = ""
            if exec_match:
                after_exec = exec_match.group(1)
                # 用 ⚠️ 分割，前为执行指令，后为风险说明
                risk_split = re.split(r'⚠️\s*风险说明[：:]', after_exec, maxsplit=1)
                exec_block = risk_split[0].strip()
                if len(risk_split) > 1:
                    risk_text = risk_split[1].strip()

            if exec_block:
                # 方向：只取“方向：”后第一个词（遇到空格、括号即停）
                dir_match = re.search(r'方向[：:]\s*([^\s(（]+)', exec_block)
                if dir_match:
                    raw_dir = dir_match.group(1).strip()
                    dir_map = {"做多": "long", "做空": "short", "观望": "neutral",
                               "long": "long", "short": "short", "neutral": "neutral"}
                    direction = dir_map.get(raw_dir, "neutral")

                # 仓位：同样只取第一个词
                pos_match = re.search(r'仓位[：:]\s*([^\s(（]+)', exec_block)
                if pos_match:
                    raw_pos = pos_match.group(1).strip()
                    pos_map = {"轻仓": "light", "中仓": "medium", "重仓": "heavy", "无": "none", "无仓位": "none"}
                    position_size = pos_map.get(raw_pos, "none")

                # 现价
                price_match = re.search(r'现价[：:]\s*([\d.]+)', exec_block)
                if price_match:
                    current_price = float(price_match.group(1))

                # 入场区间
                entry_match = re.search(r'入场区间[：:]\s*([\d.]+)\s*[-–]\s*([\d.]+)', exec_block)
                if entry_match:
                    entry_low = float(entry_match.group(1))
                    entry_high = float(entry_match.group(2))

                # 止损
                stop_match = re.search(r'止损[：:]\s*([\d.]+)', exec_block)
                if stop_match:
                    stop_loss = float(stop_match.group(1))

                # 止盈
                tp_match = re.search(r'止盈[：:]\s*([\d.]+)', exec_block)
                if tp_match:
                    take_profit = float(tp_match.group(1))

                # 说明
                plan_match = re.search(r'说明[：:]\s*(.*)', exec_block)
                if plan_match:
                    execution_plan = plan_match.group(1).strip()

            # 推翻且最终方向是观望时，清空所有价格参数
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