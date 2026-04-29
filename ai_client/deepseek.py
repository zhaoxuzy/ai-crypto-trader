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

# ------------------- 首席交易员：策略生成 -------------------
def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    timestamp = data.get("timestamp", "N/A")
    current = data['mark_price']
    above_cluster = data.get('above_cluster', 'N/A')
    below_cluster = data.get('below_cluster', 'N/A')

    if cross_symbol is None:
        if symbol == "BTC": cross_symbol = "ETH"
        elif symbol == "ETH": cross_symbol = "BTC"
        elif symbol == "SOL": cross_symbol = "BTC"
        else: cross_symbol = "ETH"

    above_trigger = "N/A"
    above_far_boundary = "N/A"
    below_trigger = "N/A"
    below_far_boundary = "N/A"

    if above_cluster != 'N/A' and '-' in above_cluster:
        parts = above_cluster.split('-')
        above_low = float(parts[0])
        above_high = float(parts[1])
        above_trigger = f"+{above_low - current:.0f}"
        above_far_boundary = f"+{above_high - current:.0f}"
    if below_cluster != 'N/A' and '-' in below_cluster:
        parts = below_cluster.split('-')
        below_low = float(parts[0])
        below_high = float(parts[1])
        below_trigger = f"-{current - below_high:.0f}"
        below_far_boundary = f"-{current - below_low:.0f}"

    data_quality = data.get("data_quality", {})
    missing = [k for k, v in data_quality.items() if v == "❌ 缺失"]
    missing_str = "、".join(missing) if missing else "无"

    max_pain = data['max_pain']
    max_pain_bias = "偏空信号" if current > max_pain else "偏多信号"
    put_call_ratio = data['put_call_ratio']
    pc_bias = "偏空信号" if put_call_ratio > 1.0 else "偏多信号"

    eth_btc_ratio = data['eth_btc_ratio']
    eth_btc_ma_7d = data.get('eth_btc_ma_7d', 0.0)
    eth_btc_percentile = data.get('eth_btc_percentile', 50.0)

    core_missing = [k for k in ["atr_15m", "above_liq", "below_liq", "cvd_slope"] if k in missing]
    constraint_note = ""
    if core_missing:
        constraint_note = f"\n【重要约束】以下核心数据缺失：{', '.join(core_missing)}。你必须将置信度设为 'low'；若清算数据缺失，则必须输出 'neutral'。\n"

    cross_context = ""
    if eth_data is not None:
        if "_complete" in eth_data:
            is_complete = eth_data["_complete"]
        else:
            crucial_fields = ['above_liq', 'below_liq', 'oi_percentile', 'funding_percentile',
                              'top_ls_percentile', 'cvd_slope', 'put_call_ratio', 'max_pain', 'mark_price']
            is_complete = all(eth_data.get(f) is not None for f in crucial_fields)
            logger.warning(f"跨币种数据缺少 _complete 标志，已自检完整性: {is_complete}")

        if not is_complete:
            cross_context = "\n【重要：跨币种数据不完整，第六步跨币种验证无法进行，对主逻辑无增强也无削弱。】\n"
        else:
            cross_current = eth_data.get('mark_price', 0)
            cross_above_liq = eth_data.get('above_liq', 0) / 1e9
            cross_below_liq = eth_data.get('below_liq', 0) / 1e9
            cross_liq_ratio = eth_data.get('liq_ratio', 0)
            cross_oi_pct = eth_data.get('oi_percentile', 50)
            cross_oi_change = eth_data.get('oi_change_24h', 0)
            cross_funding_pct = eth_data.get('funding_percentile', 50)
            cross_tls_pct = eth_data.get('top_ls_percentile', 50)
            cross_cvd = eth_data.get('cvd_slope', 0)
            cross_cvd_dir = "正（买盘）" if cross_cvd > 0 else "负（卖盘）"
            cross_context = f"""
【跨币种辅助验证数据（{cross_symbol}）】
现价：{cross_current:.2f}
清算池：上方{cross_above_liq:.2f}B(约{cross_above_liq*10:.0f}亿美元) / 下方{cross_below_liq:.2f}B(约{cross_below_liq*10:.0f}亿美元)（比值{cross_liq_ratio:.3f}）
情绪：OI分位{cross_oi_pct:.0f}%（24h{cross_oi_change:+.1f}%）、资金费率分位{cross_funding_pct:.0f}%、顶级多空比分位{cross_tls_pct:.0f}%（100%=机构极度看空）
资金流：CVD斜率{cross_cvd:.4f}（{cross_cvd_dir}）
"""

    dynamic_instructions = f"""
【时间维度动态分析】
CAUTION: CVD斜率仅代表过去，你必须结合加速度判断趋势的“续航力”。
- CVD加速度：{data.get('cvd_acceleration', 0):.3f}（正值=卖盘加速增强，负值=卖盘减弱）
- OI加速度：{data.get('oi_acceleration', 0):.3f}（正值=持仓下降加速，负值=下降减速）
- 资金费率动量：{data.get('funding_momentum', 0):.6f}
在第四步“资金流验证”的最终结论中，必须明确当前趋势属于：加速、衰竭，还是稳定。
【假突破陷阱识别】
诱导风险系数：{data.get('lure_risk_factor', 0):.2f} (>0.5 表示存在显著诱盘风险)
在第二步“猎物定位”的最终结论中，必须回答：“这个猎物是否可能是诱饵？”
若风险系数>0.5，必须给出相应的预案。
"""
    prompt = f"""你是我们加密货币交易团队的**首席交易员**，拥有十年实盘经验，管理200万U资金，精通清算动力学、多空博弈、技术分析、合约交易。请完全沉浸在这个角色中，使用第一人称（我）进行所有思考。
{constraint_note}
【{symbol} | {timestamp}】
价格：{current:.2f} | 15min ATR：{data['atr_15m']:.2f} | 1h ATR：{data.get('atr_1h', data['atr_15m']*2):.2f} | 1h波动率：{data.get('atr_1h_ratio', 0):.2f}% | 波动因子：{data['vol_factor']:.2f} | 7日分位数：{data['price_percentile']:.0f}%

清算池：
上方(空头)：{data['above_liq']/1e9:.2f}B (约{data['above_liq']/1e7:.0f}亿美元)，{above_cluster}
  触发距{above_trigger}点 (至下沿)，远边界距{above_far_boundary}点 (至上沿)
下方(多头)：{data['below_liq']/1e9:.2f}B (约{data['below_liq']/1e7:.0f}亿美元)，{below_cluster}
  触发距{below_trigger}点 (至上沿)，远边界距{below_far_boundary}点 (至下沿)
比值：{data['liq_ratio']:.3f} | 综合吸引力评分：{data.get('liquidity_bias', '中性')}（偏向上方/下方清算池）

订单簿：买{data['orderbook_bids']/1e6:.1f}M / 卖{data['orderbook_asks']/1e6:.1f}M | 失衡率{data['orderbook_imbalance']:.4f}
资金费率：{data['funding_rate']:.4f}% (分位{data['funding_percentile']:.0f}%)
OI：{data['oi']/1e9:.2f}B (约{data['oi']/1e7:.0f}亿美元) (分位{data['oi_percentile']:.0f}%)，24h{data['oi_change_24h']:+.1f}%
全市场OI：{data['agg_oi']/1e9:.2f}B，24h{data['agg_oi_change_24h']:+.1f}%
顶级多空比：{data['top_ls_ratio']:.2f} (分位{data['top_ls_percentile']:.0f}%)
恐慌贪婪：{data['fear_greed']} (7日前{data['fear_greed_prev_7d']})
期权：最大痛点{max_pain:.2f} ({max_pain_bias}) | P/C比{put_call_ratio:.4f} ({pc_bias})
资金流：CVD斜率{data['cvd_slope']:.4f} | 期货24h净流{data['netflow']/1e6:.1f}M | 交易所BTC 24h变化{data['exchange_btc_change_24h']:+.0f} BTC
ETH/BTC：当前{eth_btc_ratio:.4f}，7日均值{eth_btc_ma_7d:.4f}，7日分位数{eth_btc_percentile:.0f}%（数值越高代表ETH相对BTC越强势）
数据缺失：{missing_str}
{dynamic_instructions}
{cross_context}

【推理协议】你必须严格按此协议输出，不得偏离。每一步必须包含四个模块，但每个模块只允许用以下格式之一输出：①纯数值/符号 ②“A→B→C”因果链 ③“因为__，所以__”的压缩判断（最多10个词）。禁止任何自然语言描述性冗余。

第一步：环境定调
数据：[7日分位数：__%] [1h波动率：__%] [波动因子：__]
第一反应：分位→[超买/超卖/中性]，波动→[扩张/压缩]，因此→[适猎/待机]。
自我质疑：波动源：恐慌/贪婪？分位失效？→[是/否]。
最终结论：环境定性：[扩张且活跃→可攻击] 或 [压缩或钝化→等待]。

第二步：猎物定位
数据：[上池距/强度：__] [下池距/强度：__] [失衡率：__] [CVD：__]
第一反应：最短触发距→[上/下]侧，初始猎物。
自我质疑：有效引爆距=触发距+挂单缓冲=__，>1.5倍触发距？→[是=诱饵/否=真实]。对立池比值__→反向威胁[高/低]。裁决：真实猎物方向=[上/下]。
最终结论：第一段运动方向→[上/下]，性质→[真实猎杀/诱饵陷阱]。

第三步：对手盘解剖
数据：[OI分位：__%] [OI变化：__%] [费率分位：__%] [顶多空分位：__%]
第一反应：顶多空分位极端→[多头/空头]拥挤焦虑。
自我质疑：OI变化→[增=加注/减=投降]，费率→[加剧/反向预兆]，因此→[多杀多/空杀空]条件[成立/不成立]。
最终结论：对手盘痛苦方→[多头/空头]，被反向收割概率→[高/中/低]。

第四步：资金流验证
数据：[CVD量级/方向：__] [24h净流：__] [BTC余额变化：__]
第一反应：资金流向与价格→[同向/背离]，主推力→[真实/虚假]。
自我质疑：余额变化→长期持有者[增持/减持]，CVD斜率→[加速/衰竭]，因此动能→[增强/衰减/稳定]。
最终结论：趋势动能定性→[加速/衰竭/稳定]。

第五步：辅助信号扫描
数据：[痛点：__] [P/C：__] [ETH/BTC：__]
第一反应：痛点磁吸→[上/下/无]，P/C情绪→[顺应/对抗]主方向。
自我质疑：ETH/BTC→[确认风险偏好/分流资金]，因此辅助信号对主逻辑→[加分/中性/减分]。
最终结论：辅助信号加权→[支持/中性/警告]。

第六步：跨币种验证
数据：
① 清池方向：[BTC：__] vs [ETH：__] → [一致/矛盾]
② CVD方向：[BTC：__] vs [ETH：__] → [共振/背离]
③ 顶多空分位：BTC __% vs ETH __% → 更极端的是[__]
第一反应：两币种信号→[共振强化/背离冲突]。
自我质疑：若背离，是结构性分流还是风险传导？→因__，所以单币种需修正为__。
最终结论：修正后单币种置信度→[原级/降级]，新增风险点→[__]。

第七步：制定交易计划
价格路径推演：流动性猎杀→[先扫__后反杀__]，对手盘__方被挤压，CVD__向确认，路径→[__]。
最终策略：
- 币种：{symbol}
- 方向：[做多/做空/观望]
- 现价：
- 仓位：[轻/中/重]
- 置信度：[高/中/低]
- 入场区间：[__-__] （依据：__）
- 止损：[__] （依据：__）
- 止盈：[__] （依据：__）
- 说明：[__]
主动证伪：[开盘__分钟内若__发生，则否定策略]
微观确认：[入场需见__盘口信号]

输出JSON（不要代码块）：
{{
  "direction": "做多 / 做空 / 观望",
  "confidence": "高 / 中 / 低",
  "position_size": "轻仓 / 中仓 / 重仓 / 无",
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "execution_plan": "一句话指令",
  "reasoning": "第一步到第七步的完整推演文本，包含所有分析、第一反应、自我质疑、最终结论",
  "risk_note": "风险说明",
  "final_strategy": "请将第七步'最终合约策略'的标准格式文本（包含币种、方向、现价、仓位、置信度、入场区间、止损、止盈、说明、主动证伪、微观确认）完整填入此字段。格式如下：\n- 币种：{symbol}\n- 方向：[做多/做空/观望]\n- 现价：\n- 仓位：[轻/中/重]\n- 置信度：[高/中/低]\n- 入场区间：[__-__] （依据：__）\n- 止损：[__] （依据：__）\n- 止盈：[__] （依据：__）\n- 说明：[__]\n主动证伪：[__]\n微观确认：[__]"
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
            logger.info(f"DeepSeek 调用 (尝试 {attempt+1}/{max_retries})")
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
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

            # ========== 标准化 A 的输出 ==========
            # 方向标准化（兼容中英文）
            dir_map = {"做多": "long", "做空": "short", "观望": "neutral",
                       "long": "long", "short": "short", "neutral": "neutral"}
            raw_dir = s.get("direction", "")
            if raw_dir in dir_map:
                s["direction"] = dir_map[raw_dir]
            else:
                # 尝试从 reasoning 的最终策略文本中提取
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
            return s

        except Exception as e:
            logger.warning(f"调用失败: {e}")
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
    old_risk = s.get('risk_note', '')
    s["direction"] = "neutral"
    s["confidence"] = "low"
    s["position_size"] = "none"
    s["entry_price_low"] = 0
    s["entry_price_high"] = 0
    s["stop_loss"] = 0
    s["take_profit"] = 0
    s["execution_plan"] = ""
    s["reasoning"] = (s.get("reasoning", "") + f"\n\n[原始信号因校验规则被强制改为观望，原因：{reason}").strip()
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

        if above_liq <= 0 and below_liq <= 0 and direction != "neutral":
            _force_neutral(s, "清算数据缺失，强制输出 neutral")
            return True, "已自动修正为观望"

        if atr_15m <= 0 and s.get("confidence") == "high":
            s["confidence"] = "medium"
            logger.warning("核心数据缺失(atr_15m)，置信度强制降级为 medium")
        if cvd_slope is None and s.get("confidence") == "high":
            s["confidence"] = "medium"
            logger.warning("核心数据缺失(cvd_slope)，置信度强制降级为 medium")

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
        # 继续校验，不返回False

    if direction == "long" and stop_loss >= entry_low:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间下方，请人工确认。"
    elif direction == "short" and stop_loss <= entry_high:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间上方，请人工确认。"

    try:
        if direction == "long":
            worst_entry = entry_high
            risk = worst_entry - stop_loss
            reward = take_profit - worst_entry
        else:
            worst_entry = entry_low
            risk = stop_loss - worst_entry
            reward = worst_entry - take_profit

        if risk > 0:
            s["_calculated_rr"] = round(reward / risk, 2)
        else:
            s["_calculated_rr"] = 0.0
    except Exception as e:
        logger.warning(f"计算盈亏比时出错: {e}")
        s["_calculated_rr"] = None

    return True, ""

# ------------------- 风控审计官：逻辑审计 -------------------
def build_reviewer_prompt(original_strategy: dict, data: dict, symbol: str) -> str:
    liquidity_bias = data.get('liquidity_bias', 'neutral')
    bias_map = {'long': '偏向上方', 'short': '偏向下止', 'neutral': '无明确偏向'}
    bias_text = bias_map.get(liquidity_bias, '无数据')

    return f"""你是加密货币交易团队的风控审计官，唯一任务是在首席交易员提交的策略中找出可能存在的遗漏、矛盾或与数据不符之处。

【交易标的】{symbol}
【代码层客观锚点】清算池综合吸引力评分：{bias_text}（基于规模/触发距/订单簿计算，数学模型得出）
【原策略方向】{original_strategy.get('direction')}
【入场/止损/止盈】{original_strategy.get('entry_price_low')} - {original_strategy.get('entry_price_high')} / {original_strategy.get('stop_loss')} / {original_strategy.get('take_profit')}

【原策略推演过程】
{original_strategy.get('reasoning', '无推演过程')}

【审查要求】请严格按照以下模板输出

【风控审计官 - 审计报告】
一、遗漏指标与分析缺失：
[若有遗漏，按此格式：在[步骤X/决策点]中，交易员未分析已提供的[指标名称/数据项]。该指标显示[具体数值/信号]，若纳入分析将[强化/削弱/推翻]当前方向判断。 [严重性：高/中/低]]
[若无遗漏，写“已覆盖所有应分析的关键指标”]
二、数据与解读错误：
[若有错误，按此格式：在[步骤X]中，交易员声称[数值/解读]，但实际数据为[数值/正确含义]。此错误[是否影响方向判断]。 [严重性：高/中/低]]
[若无，写“未发现数据或解读错误”]
三、逻辑错误：
[若有错误，按此格式：[错误类型]在[步骤X]：[描述]。 [严重性：高/中/低]]
[若无，写“未发现明显逻辑错误”]
四、关键反证提示：
[若原策略推演中引用的数据或逻辑，与已提供的其他数据存在明显矛盾，按此格式指出：在[步骤X]中，策略依据[数据A]得出[结论]，但已提供的[数据B]显示[相反信号]，二者构成矛盾，未被交易员处理。 [严重性：高/中/低]]
[若无矛盾，写“未发现关键反证被忽略”]
五、博弈层面审视：
-基于已提供的清算池评分，做市商更可能向哪个方向猎杀流动性？策略的止损位是否暴露在该路径上？（若数据不足以判断，写明“数据不足，无法判定”）
-预设的入场区间，是否与已提供图表中的关键结构位（前高/前低/成交密集区）重合，从而可能成为对手盘的流动性来源？（若数据不足以判断，写明“数据不足，无法判定”）
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
            logger.info(f"风控审计官 调用 (尝试 {attempt+1}/{MAX_RETRIES})")
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                timeout=120
            )
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
                
            return {"verdict": verdict, "full_report": content, "severity_counts": severity_counts}
        except Exception as e:
            logger.warning(f"风控审计官调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                logger.warning("风控审计官所有重试均失败，标记为通过，跳过审计")
                return {"verdict": "通过", "full_report": "审计官调用失败，跳过审计"}

# ------------------- 交易委员会：最终裁决 -------------------
def build_judge_prompt(original_strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> str:
    orig_dir = original_strategy.get('direction', 'neutral')
    entry_l = original_strategy.get('entry_price_low', 'N/A')
    entry_h = original_strategy.get('entry_price_high', 'N/A')
    stop_l = original_strategy.get('stop_loss', 'N/A')
    tp_l = original_strategy.get('take_profit', 'N/A')
    
    report = reviewer_report.get('full_report', '无审计报告')

    # 只提取关键字段，不再传完整JSON
    key_data = {
        "mark_price": data.get("mark_price"),
        "liquidity_bias": data.get("liquidity_bias"),
        "above_liq": f"{data.get('above_liq', 0)/1e9:.2f}B",
        "below_liq": f"{data.get('below_liq', 0)/1e9:.2f}B",
        "above_trigger": data.get("above_trigger"),
        "below_trigger": data.get("below_trigger"),
        "liq_ratio": data.get("liq_ratio"),
        "cvd_slope": data.get("cvd_slope"),
        "cvd_acceleration": data.get("cvd_acceleration"),
        "oi_percentile": data.get("oi_percentile"),
        "top_ls_percentile": data.get("top_ls_percentile"),
        "funding_rate": data.get("funding_rate"),
        "netflow_24h": data.get("netflow"),
        "atr_15m": data.get("atr_15m"),
        "max_pain": data.get("max_pain"),
        "orderbook_imbalance": data.get("orderbook_imbalance"),
    }

    prompt = f"""你是最终决策的**独立交易委员会主席**，拥有二十年加密货币短线合约交易经验。
你的职责是：基于【市场数据】，公正审核【首席交易员的策略】和【风控审计报告】，给出最合理的执行方案。

【市场数据】
交易标的：{symbol}
（以下为关键指标快照，所有判断必须引用这些字段）
{json.dumps(key_data, ensure_ascii=False, indent=2)}

【原策略】方向：{orig_dir}，入场：{entry_l}-{entry_h}，止损：{stop_l}，止盈：{tp_l}
【推演过程】{original_strategy.get('reasoning', '无')[:1000]}
【审计报告】{report}
【裁决规则】
1.事实为上：所有判断必须引用【市场数据】的具体字段和数值，不得凭感觉或记忆。
2.独立公正：既不听信首席交易员的一面之词，也不盲从审计官的指控。你必须亲自核验每一项审计指出的错误，同时也要主动检查审计官是否遗漏了重要反证。
3.逻辑自洽：你的最终结论（方向、仓位、价格）必须与你列出的核验依据形成闭环，不能自相矛盾。
4.双向核验与反证风险评估
(1) 对审计报告中的每一条指控，完成以下操作：
  a) 找到该指控引用的市场数据字段，核对数值是否一致。
  b) 若数值不一致 → 直接驳回该指控，并说明审计官的错误。
  c) 若数值一致 → 判断该错误是否实质影响交易方向。若不影响方向，可标记为“部分采纳”或“驳回”。
  d) 反证风险评估（高/中/低）：你必须提出至少一条可能挑战你裁决结论的证据。反证必须包含：该证据是什么、它为何可能构成威胁、以及你为何最终排除它。若你搜寻了所有数据后确实认为毫无矛盾，才可填写“无”，但必须简述你的搜寻范围作为理由。
(2) 同时，你要主动检查【市场数据】中是否存在审计官未提及但可能影响方向的信号（例如，审计官漏掉了重要的反向数据），若有，必须在裁决理由中补充并同样进行反证风险评估。
5.输出最终决策要求
  a) 综合核验结果，判断是否维持原策略方向。
  b) 若你决定推翻原方向，必须满足：
      1) 明确指出原方向依赖的一个核心假设，并用市场数据字段证伪。
      2) 给出至少两个支持新方向的独立数据字段（字段名+数值）。
      3) 新方向必须与清算池综合吸引力评分（liquidity_bias）在逻辑上自洽，若矛盾必须给出强证据链解释，否则必须采纳该锚点。
  c) 所有入场、止损、止盈价格必须基于ATR、清算池边界或期权关键位等客观数据，禁止凭感觉设定。

【输出模板】：必须按照以下模板输出完整内容。  

📋 裁决说明：
   -在此处逐条列出对审计指控的裁决，每条必须包含：指控内容、裁决(采纳/驳回)、依据(说明依据，引用具体数据)
      1. 指控内容：[原文概括]
       裁决结论：采纳/驳回/部分采纳
       核验依据：（引用字段+数值）
       反证风险评估：（高/中/低，证据及排除理由）
      2.
  -核心逻辑：说明裁决后制定的策略逻辑，必须提供站的住脚的证据或推论。
📌 最终判决：[维持原判 / 修正参数 / 降级执行 / 推翻]
🎯 执行指令（基于裁决结果，结合交易员策略和审计结论，制定交易策略，**必须按照以下格式输出**，否则视为无效）：
   - 币种：{symbol}
   - 方向：[做多 / 做空 / 观望]
   - 现价：
   - 仓位：[轻仓 / 中仓 / 重仓 / 无]
   - 入场区间：[价格下限-价格上限]（说明依据，若观望则写“无”）
   - 止损：[价格]（说明依据，若观望则写“无”）
   - 止盈：[价格]（说明依据，若观望则写“无”）
   - 说明：[一句话指令，或观望时的触发条件]
⚠️ 风险说明：[列出关键风险及应对措施]
"""
    return prompt


def call_judge(original_strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> dict:
    prompt = build_judge_prompt(original_strategy, reviewer_report, data, symbol)
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=TIMEOUT_SECONDS
    )
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"交易委员会 调用 (尝试 {attempt+1}/{MAX_RETRIES})")
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                timeout=120
            )
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip():
                raise ValueError("交易委员会响应为空")

            # ---------- 采集执行指令块 ----------
            original_dir = original_strategy.get("direction", "neutral")
            direction = original_dir
            position_size = original_strategy.get("position_size", "none")
            entry_low = original_strategy.get("entry_price_low", 0)
            entry_high = original_strategy.get("entry_price_high", 0)
            stop_loss = original_strategy.get("stop_loss", 0)
            take_profit = original_strategy.get("take_profit", 0)
            execution_plan = ""
            current_price = data.get("mark_price", 0)

            # 提取执行指令块
            exec_section = re.search(r'🎯\s*执行指令[：:]?\s*(.*?)(?=📋|⚠️|$)', content, re.DOTALL)
            if exec_section:
                exec_text = exec_section.group(1).strip()

                # 1. 方向解析并标准化
                dir_match = re.search(r'方向[：:]\s*(做多|做空|观望|long|short|neutral)', exec_text)
                if dir_match:
                    raw_dir = dir_match.group(1)
                    dir_map = {"做多": "long", "做空": "short", "观望": "neutral",
                               "long": "long", "short": "short", "neutral": "neutral"}
                    direction = dir_map.get(raw_dir, original_dir)

                # 2. 仓位解析并标准化
                pos_match = re.search(r'仓位[：:]\s*(轻仓|中仓|重仓|无|无仓位|light|medium|heavy|none)', exec_text)
                if pos_match:
                    raw_pos = pos_match.group(1)
                    pos_map = {"轻仓": "light", "中仓": "medium", "重仓": "heavy", "无": "none", "无仓位": "none",
                               "light": "light", "medium": "medium", "heavy": "heavy", "none": "none"}
                    position_size = pos_map.get(raw_pos, position_size)

                # 3-6. 价格提取
                price_match = re.search(r'现价[：:]\s*([\d.]+)', exec_text)
                if price_match:
                    current_price = float(price_match.group(1))

                entry_match = re.search(r'入场区间[：:]\s*([\d.]+)\s*[-–]\s*([\d.]+)', exec_text)
                if entry_match:
                    entry_low = float(entry_match.group(1))
                    entry_high = float(entry_match.group(2))

                stop_match = re.search(r'止损[：:]\s*([\d.]+)', exec_text)
                if stop_match:
                    stop_loss = float(stop_match.group(1))

                tp_match = re.search(r'止盈[：:]\s*([\d.]+)', exec_text)
                if tp_match:
                    take_profit = float(tp_match.group(1))

                # 7. 说明
                plan_match = re.search(r'说明[：:]\s*(.*)', exec_text)
                if plan_match:
                    execution_plan = plan_match.group(1).strip()
                else:
                    execution_plan = exec_text.replace('\n', ' ').strip()

            # 确定判决类型：直接根据最终方向与原方向的关系，结合判决文本
            verdict_match = re.search(r'📌\s*最终判决[：:]\s*(.*)', content)
            verdict_text = verdict_match.group(1).strip() if verdict_match else ""
            verdict_text_clean = re.sub(r'\*{1,2}', '', verdict_text).strip()

            if direction == "neutral" and direction != original_dir:
                verdict = "推翻改为观望"
                entry_low, entry_high, stop_loss, take_profit = 0, 0, 0, 0
                position_size = "none"
            elif direction != original_dir and direction != "neutral":
                verdict = "推翻改为反向操作"
            elif "修正" in verdict_text_clean:
                verdict = "修正参数"
            elif "降级" in verdict_text_clean:
                verdict = "降级执行"
            elif "推翻" in verdict_text_clean:
                verdict = "推翻改为观望"
                direction = "neutral"
                entry_low, entry_high, stop_loss, take_profit = 0, 0, 0, 0
                position_size = "none"
            else:
                verdict = "维持原判"

            # 提取裁决理由和风险说明
            reasoning_block = ""
            reason_section = re.search(r'📋\s*(裁决说明|裁决理由)[：:]?\s*(.*?)(?=⚠️|$)', content, re.DOTALL)
            if reason_section:
                reasoning_block = reason_section.group(0).strip()

            risk_match = re.search(r'⚠️\s*风险说明[：:]\s*(.*)', content, re.DOTALL)
            risk_block = risk_match.group(0).strip() if risk_match else ""
            risk_note = risk_match.group(1).strip() if risk_match else ""

            judge_result = {
                "judge_C": {
                    "final_verdict": verdict,
                    "verdict_level": "A",
                    "final_direction": direction,
                    "final_confidence": "medium",
                    "final_position_size": position_size,
                    "entry_price_low": entry_low,
                    "entry_price_high": entry_high,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "execution_plan": execution_plan,
                    "reasoning": content,
                    "risk_note": risk_note,
                    "title_line": verdict_match.group(0).strip() if verdict_match else "",
                    "exec_block": exec_section.group(0).strip() if exec_section else "",
                    "reasoning_block": reasoning_block,
                    "risk_block": risk_block,
                    "current_price": current_price
                }
            }

            logger.info(f"交易委员会裁决: {verdict}, 方向: {direction}")
            return judge_result

        except Exception as e:
            logger.warning(f"交易委员会调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                return {"judge_C": {"final_verdict": "维持原判", "verdict_level": "A"}}


def _validate_execution_direction(s: dict):
    exec_plan = s.get("execution_plan", "")
    final_direction = s.get("direction", "")
    if final_direction == "long" and "做空" in exec_plan and "做多" not in exec_plan:
        logger.warning("输出矛盾：方向为long但执行指令包含做空且未提及做多，已自动将方向改为neutral")
        _force_neutral(s, "输出矛盾：方向与执行指令不一致")
    elif final_direction == "short" and "做多" in exec_plan and "做空" not in exec_plan:
        logger.warning("输出矛盾：方向为short但执行指令包含做多且未提及做空，已自动将方向改为neutral")
        _force_neutral(s, "输出矛盾：方向与执行指令不一致")


def apply_final_verdict(original_strategy: dict, judge_result: dict, reviewer_report: dict = None) -> dict:
    verdict = judge_result.get("judge_C", {}).get("final_verdict", "维持原判")
    final = judge_result.get("judge_C", {})

    logger.info(f"应用最终决议: {verdict}")

    original_strategy["_reviewed"] = True
    original_strategy["_original_direction"] = original_strategy.get("direction")
    original_strategy["_review_verdict"] = verdict
    original_strategy["_judge_data"] = final

    original_strategy["_title_line"] = final.get("title_line", "")
    original_strategy["_exec_block_raw"] = final.get("exec_block", "")
    original_strategy["_reasoning_block_raw"] = final.get("reasoning_block", "")
    original_strategy["_risk_block_raw"] = final.get("risk_block", "")

    def _clean_reasoning(raw):
        if not raw:
            return ""
        if "审计指控" in raw:
            return raw[raw.find("审计指控"):].strip()
        if "📋 裁决理由" in raw:
            return raw[raw.find("📋 裁决理由") + len("📋 裁决理由"):].strip()
        return raw.strip()

    if verdict == "维持原判":
        original_strategy["_judge_reasoning"] = _clean_reasoning(final.get("reasoning", ""))
    elif verdict == "修正参数":
        original_strategy["direction"] = final.get("final_direction", original_strategy.get("direction"))
        original_strategy["confidence"] = final.get("final_confidence", original_strategy.get("confidence"))
        original_strategy["position_size"] = final.get("final_position_size", original_strategy.get("position_size"))
        original_strategy["entry_price_low"] = final.get("entry_price_low", original_strategy["entry_price_low"])
        original_strategy["entry_price_high"] = final.get("entry_price_high", original_strategy["entry_price_high"])
        original_strategy["stop_loss"] = final.get("stop_loss", original_strategy["stop_loss"])
        original_strategy["take_profit"] = final.get("take_profit", original_strategy["take_profit"])
        original_strategy["execution_plan"] = final.get("execution_plan", original_strategy.get("execution_plan", ""))
        original_strategy["risk_note"] = final.get("risk_note", original_strategy.get("risk_note", ""))
        original_strategy["_judge_reasoning"] = _clean_reasoning(final.get("reasoning", ""))
    elif verdict == "降级执行":
        original_strategy["direction"] = final.get("final_direction", original_strategy.get("direction"))
        original_strategy["confidence"] = final.get("final_confidence", original_strategy.get("confidence"))
        original_strategy["entry_price_low"] = final.get("entry_price_low", original_strategy["entry_price_low"])
        original_strategy["entry_price_high"] = final.get("entry_price_high", original_strategy["entry_price_high"])
        original_strategy["stop_loss"] = final.get("stop_loss", original_strategy["stop_loss"])
        original_strategy["take_profit"] = final.get("take_profit", original_strategy["take_profit"])
        original_strategy["execution_plan"] = final.get("execution_plan", original_strategy.get("execution_plan", ""))
        original_strategy["risk_note"] = final.get("risk_note", original_strategy.get("risk_note", ""))
        size_map = {"heavy": "medium", "medium": "light", "light": "light"}
        original_strategy["position_size"] = size_map.get(final.get("final_position_size", original_strategy.get("position_size", "light")), "light")
        original_strategy["_judge_reasoning"] = _clean_reasoning(final.get("reasoning", ""))
    elif verdict == "推翻改为观望":
        _force_neutral(original_strategy, f"交易委员会决议: {verdict}")
        original_strategy["_judge_reasoning"] = _clean_reasoning(final.get("reasoning", ""))
    elif verdict == "推翻改为反向操作":
        original_strategy["direction"] = final.get("final_direction", original_strategy.get("direction"))
        original_strategy["confidence"] = final.get("final_confidence", original_strategy.get("confidence"))
        original_strategy["position_size"] = final.get("final_position_size", original_strategy.get("position_size"))
        original_strategy["entry_price_low"] = final.get("entry_price_low", 0)
        original_strategy["entry_price_high"] = final.get("entry_price_high", 0)
        original_strategy["stop_loss"] = final.get("stop_loss", 0)
        original_strategy["take_profit"] = final.get("take_profit", 0)
        original_strategy["execution_plan"] = final.get("execution_plan", "")
        original_strategy["risk_note"] = final.get("risk_note", "")
        original_strategy["_judge_reasoning"] = _clean_reasoning(final.get("reasoning", ""))

    # 最终安全检查：neutral 时强制清空交易参数
    if original_strategy.get("direction") == "neutral":
        original_strategy["entry_price_low"] = 0
        original_strategy["entry_price_high"] = 0
        original_strategy["stop_loss"] = 0
        original_strategy["take_profit"] = 0
        original_strategy["position_size"] = "none"
        if not original_strategy.get("execution_plan"):
            original_strategy["execution_plan"] = "观望"

    _validate_execution_direction(original_strategy)
    return original_strategy
