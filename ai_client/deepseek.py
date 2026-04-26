import os
import json
import time
import re
from datetime import datetime
from openai import OpenAI
from utils.logger import logger

TICK_SIZE = 0.1
MAX_RETRIES = 2
RETRY_BASE_WAIT = 2
TIMEOUT_SECONDS = 180

def build_prompt(data: dict, symbol: str, eth_data: dict = None, cross_symbol: str = None) -> str:
    timestamp = data.get("timestamp", "N/A")
    current = data['mark_price']
    above_cluster = data.get('above_cluster', 'N/A')
    below_cluster = data.get('below_cluster', 'N/A')

    if cross_symbol is None:
        if symbol == "BTC":
            cross_symbol = "ETH"
        elif symbol == "ETH":
            cross_symbol = "BTC"
        elif symbol == "SOL":
            cross_symbol = "BTC"
        else:
            cross_symbol = "ETH"

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
若风险系数>0.5，必须在第七步“假突破预设”中给出明确的应激方案。
"""

    prompt = f"""你是一个拥有十年经验管理200万U的顶尖加密货币短线交易员，精通清算动力学、多空博弈、技术分析、合约交易。请完全沉浸在这个角色中，使用第一人称（我）进行所有思考。
{constraint_note}
【{symbol} | {timestamp}】
价格：{current:.2f} | 15min ATR：{data['atr_15m']:.2f} | 1h ATR：{data.get('atr_1h', data['atr_15m']*2):.2f} | 1h波动率：{data.get('atr_1h_ratio', 0):.2f}% | 波动因子：{data['vol_factor']:.2f} | 7日分位数：{data['price_percentile']:.0f}%

清算池：
上方(空头)：{data['above_liq']/1e9:.2f}B (约{data['above_liq']/1e7:.0f}亿美元)，{above_cluster}
  触发距{above_trigger}点 (至下沿)，远边界距{above_far_boundary}点 (至上沿)
下方(多头)：{data['below_liq']/1e9:.2f}B (约{data['below_liq']/1e7:.0f}亿美元)，{below_cluster}
  触发距{below_trigger}点 (至上沿)，远边界距{below_far_boundary}点 (至下沿)
比值：{data['liq_ratio']:.3f}

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
---
【硬性约束】
1. 必须且只能引用上方提供的具体数据，不得编造、估算或使用记忆中的任何数值。
2. 你的 `reasoning` 字段必须包含从第一步到第九步的完整推演文本，每一步都必须显式包含“分析数据”、“第一反应”、“自我质疑”、“最终结论”子标题及详细完整内容。
---
第一步：环境定调
分析数据：价格7日分位数、1h波动率、波动因子。
第一反应：
自我质疑：
最终结论：

第二步：猎物定位
分析数据：上下方清算池距离/强度、比值、订单簿买卖盘量、失衡率。
定位规则：“猎物距离”必须是触发距，即价格到清算区最近边界的距离。价格仅需触碰该边界，即可引爆该侧流动性。
推演要求：你需要基于上述规则，独立完成对第一段运动方向和幅度的战术推演。推演应围绕以下维度展开，最终自然得出你的判断：
- 攻击成本：基于触发距，哪个方向的猎物是阻力最小的路径？
- 盘口阻力：订单簿失衡率和CVD斜率是在助推还是阻挠该方向？如果两者矛盾，你需要做出裁决并说明依据。
- 反向张力：对立侧清算池的规模和比值是否构成反向威胁？在什么情况下它会改变你的方向判断？
诱饵审视：你必须对当前猎物的真实性进行双向审视：一方面，它是否是做市商设置的危险陷阱，把明显的一侧作为诱饵来反手猎杀另一方？另一方面，如果它不是诱饵，而是真实的流动方向，市场会以怎样的方式确认这一点？你的推演必须包含这两种可能性，并在最终的策略中体现你对这个分歧的裁决。
第一反应：
自我质疑：
最终结论：

第三步：对手盘解剖
分析数据：OI分位数及变化、全市场OI变化、资金费率分位数、顶级多空比分位数、恐慌贪婪及趋势。
第一反应：
自我质疑：
最终结论：

第四步：资金流验证
分析数据：CVD斜率方向/量级、期货24h净流、交易所BTC余额变化。
【趋势动能研判】在最终结论中必须明确当前趋势是加速、衰竭还是稳定。
第一反应：
自我质疑：
最终结论：

第五步：辅助信号扫描
分析数据：期权最大痛点、P/C比、ETH/BTC汇率。
第一反应：
自我质疑：
最终结论：

第六步：跨币种验证
分析数据：必须完成以下三项强制对比，每项写出具体数值，基于此三组客观对比，深度分析两币种整体方向是一致还是矛盾，并解释共振或背离的市场含义.
① 清算池比值方向对比：[ ] vs [ ]，两者方向是否一致？
② CVD斜率方向对比：[ ] vs [ ]，资金流向是否共振？
③ 顶级多空比分位对比：[ ]% vs [ ]%，哪个币种的机构空头更极端？
第一反应：
自我质疑：
最终结论：
跨币种反馈：基于以上对比，我的单币种结论被如何修正？有无新增风险点？如果有，我必须回到第三步或第五步的自我质疑中重新审视。

第七步：交叉验证与裁决【你刚刚完成了对市场所有维度的扫描。现在，你面对的可能是一个多空交织的矛盾体。请基于前六步的所有数据和结论，独立完成以下裁决。你的回答必须明确包含最终方向、置信度和仓位】
信号传唤：
   - 明确指出前六步中，哪一个信号是你见过最强的看涨证据？哪一个信号是你见过最强的看跌证据？
权重审判：
   - 为你做最终裁决所依据的关键决策因子分配权重（合计100%）。
   - 必须简要说明：在当前的市场阶段（是处于趋势的早期、末期，还是震荡中），为何你会赋予某些因子压倒性的权重，而让另一些因子近乎失效？
心证交锋：
   - 你必须模拟你的内心对话：扮演一个与你的初步裁决方向完全相反的交易员，用前六步中最强的反向证据作为武器，攻击你的逻辑。
   - 然后，你再以主交易员的身份，驳斥或接纳这个攻击。
   - 最终解释，你为何在当前的特定时间、特定价位下，选择压过最强反向信号的干扰。若你被反向证据说服，你必须立即改变立场，选择观望。
核心假设：
   - 清晰写出你判断方向的核心逻辑链条。它不是事实的堆砌，而是你脑海中的那份战术蓝图：“因为[关键假设A]，所以会出现[现象B]，进而引发[C的行情结果]。”
证伪条件：
   - 设定你的“逃生门”：具体的价格行为，以及必须同时发生的资金流或订单簿确认。
   - 声明：仅有价格穿刺而无资金流确认的突破，将被我视为陷阱。两者必须共同触发，我的核心假设才被推翻，我将无条件执行撤离。

第八步：交易执行计划【基于第七步的最后裁决结果，制定可直接执行的具体计划】
价格路径推演：必须综合运用流动性猎杀理论（清算池位置/强度）、行为金融学（对手盘心理、恐慌/贪婪、顶级多空比极端值）以及博弈论（做市商与散户的短期博弈策略）进行推演，价格最可能如何测试并触发关键流动性区域？触发后会产生何种连锁强平或踩踏？
合约策略：入场区间+止损位+止盈位，说明具体的依据。
主动证伪信号：
微观盘口确认：

第九步：推理自检【在输出最终JSON之前，请逐条回答以下问题，不得跳过任何一题】
1.我的最终裁决是否完全基于前六步的数据和结论？
2.我在哪一步的“自我质疑”中发现了后来被证实为关键的风险点？
3.如果我错了，最可能是在哪一步的假设上栽了跟头？
4.我是否过于武断地压低了某个反向信号的权重？具体是哪一个信号？如果该反向信号最终被证明是正确的，我会犯下怎样的错误？

【有效阈值计算指令】
`decision_summary` 中的 `effective_threshold` 必须按 `max(0.8 * 1h_ATR, 当前价格 * 0.05%)` 计算，`threshold_rationale` 中需写明该计算式及你对此阈值在当前市况中的看法。

{{
  "decision_summary": {{
    "final_direction": "看涨/看跌/观望",
    "first_leg_direction": "上涨/下跌",
    "first_leg_magnitude": 0.0,
    "effective_threshold": 0.0,
    "threshold_rationale": "我选择此阈值的逻辑：...",
    "chosen_action": "短线做多/短线做空/观望"
  }},
  "direction": "long/short/neutral",
  "confidence": "high/medium/low",
  "position_size": "light/medium/heavy/none",
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "execution_plan": "一句话指令",
  "reasoning": "第一步到第九步的完整推演文本",
  "risk_note": "风险说明"
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
                max_tokens=32768,
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

            s.setdefault("position_size", "none")
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
    s["reasoning"] += f"\n\n[系统自动干预] 原始信号因校验规则被强制改为观望。原因：{reason}"
    s["risk_note"] = f"[系统干预] 原始方向因校验规则被强制改为观望。" + (f" 原始风险说明：{old_risk}" if old_risk else "")


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
        return False, "入场区间下限大于上限"

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

    summary = s.get("decision_summary", {})
    if summary:
        final_dir = summary.get("final_direction", "")
        if final_dir == "看涨" and direction != "long":
            _force_neutral(s, f"decision_summary最终方向为看涨，但direction为{direction}")
            return True, "已自动修正为观望"
        if final_dir == "看跌" and direction != "short":
            _force_neutral(s, f"decision_summary最终方向为看跌，但direction为{direction}")
            return True, "已自动修正为观望"
        if final_dir == "观望" and direction != "neutral":
            _force_neutral(s, f"decision_summary最终方向为观望，但direction为{direction}")
            return True, "已自动修正为观望"

    return True, ""


def build_reviewer_prompt(original_strategy: dict, data: dict, symbol: str) -> str:
    return f"""你是一位顶级加密货币交易策略的审查官。你的使命是对以下策略进行逐项审查，找出逻辑漏洞、数据曲解和思考盲点。你不给出最终方向，只输出结构化的审查报告。

【交易标的】{symbol}
【市场数据】（这是策略制定时所依据的全部数据，审查时请严格对照）
{json.dumps(data, ensure_ascii=False, indent=2)}

【原策略裁决】
方向：{original_strategy.get('direction')}
置信度：{original_strategy.get('confidence')}
仓位：{original_strategy.get('position_size')}
入场：{original_strategy.get('entry_price_low')} - {original_strategy.get('entry_price_high')}
止损：{original_strategy.get('stop_loss')}
止盈：{original_strategy.get('take_profit')}

【原策略完整推演过程】（必须与上方市场数据逐一核对）
{original_strategy.get('reasoning', '无推演过程')}

【审查要求】
你必须对策略的每一步进行审查，使用“通过/存疑/驳回”三级标注，并附上严重性权重（轻度/中度/重大）和证据链接。

输出JSON（不要代码块）：
{{
  "reviewer_B": {{
    "step_by_step": [
      {{
        "step": "第一步：环境定调",
        "verdict": "通过/存疑/驳回",
        "severity": "轻度/中度/重大",
        "issue": "问题描述（若通过则留空）",
        "evidence": "证据（引用原推理或数据）",
        "suggestion": "修正建议"
      }},
      ... 共九步
    ],
    "summary_severity": "轻度/中度/重大",
    "overall_issues": ["问题1", "问题2", ...]
  }}
}}
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
            logger.info(f"审查官B 调用 (尝试 {attempt+1}/{MAX_RETRIES})")
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                timeout=TIMEOUT_SECONDS
            )
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip():
                raise ValueError("审查官响应为空")
            json_str = extract_json(content)
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"审查官B调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                raise
    return {}


def build_judge_prompt(original_strategy: dict, reviewer_report: dict, data: dict, symbol: str) -> str:
    return f"""你是一位顶级加密货币交易策略的终审法官。你将收到交易员A的策略和审查官B的审查报告。你的使命是做出最终裁决。

【交易标的】{symbol}
【市场数据】
{json.dumps(data, ensure_ascii=False, indent=2)}

【交易员A的策略】
{json.dumps(original_strategy, ensure_ascii=False, indent=2)}

【审查官B的审查报告】
{json.dumps(reviewer_report, ensure_ascii=False, indent=2)}

【裁决流程】
你必须严格按照以下流程图进行裁决，不得跳过任何步骤：

1. 检查审查报告中是否有“驳回”项。若无驳回 → 维持原判（A级）。
2. 若有驳回，检查A的推演中是否已提前回应并有效驳斥了该驳回点。若已有效回应 → 该驳回无效。
3. 若存在有效驳回，判断该问题是否可以通过修正参数（入场、止损、止盈）解决。如果可以 → 修正参数（B级）。
4. 若问题无法通过修正参数解决，判断问题的严重程度。
   - 轻度/中度 → 降级执行（C级），将原仓位降一级（heavy→medium→light）。
   - 重大 → 推翻改为观望（E级）。

【例外条款】
如果你认为严格按照上述流程图会导致明显违背市场逻辑的结果，你可以绕过流程，但必须同时满足以下三个条件：
1. 在裁决书中显式声明：“[例外触发] 我选择不遵循标准流程图，因为……（列出具体原因，必须引用数据）”；
2. 扮演一个与你最终裁决方向相反的交易员，用流程图规定的方式攻击你的决定，并展示你如何驳斥它；
3. 再次确认：最终裁决是否仍然比按流程图执行更优？

输出JSON（不要代码块）：
{{
  "judge_C": {{
    "final_verdict": "维持原判/修正参数/降级执行/补充条件执行/推翻改为观望",
    "verdict_level": "A/B/C/D/E",
    "exception_used": "是/否",
    "exception_reason": "若使用例外条款，填写原因",
    "final_direction": "long/short/neutral",
    "final_confidence": "high/medium/low",
    "final_position_size": "light/medium/heavy/none",
    "entry_price_low": 0.0,
    "entry_price_high": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "execution_plan": "一句话指令",
    "reasoning": "你的完整裁决过程，包括对每个驳回项的回应",
    "risk_note": "风险说明"
  }}
}}
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
            logger.info(f"终审法官C 调用 (尝试 {attempt+1}/{MAX_RETRIES})")
            resp = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                timeout=TIMEOUT_SECONDS
            )
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip():
                raise ValueError("法官响应为空")
            json_str = extract_json(content)
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"法官C调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
            else:
                raise
    return {}


def apply_final_verdict(original_strategy: dict, judge_result: dict) -> dict:
    verdict = judge_result.get("judge_C", {}).get("final_verdict", "维持原判")
    final = judge_result.get("judge_C", {})

    original_strategy["_reviewed"] = True
    original_strategy["_original_direction"] = original_strategy.get("direction")
    original_strategy["_review_verdict"] = verdict
    original_strategy["_review_issues"] = final.get("overall_issues", [])

    if verdict == "维持原判":
        return original_strategy

    elif verdict == "修正参数":
        original_strategy["entry_price_low"] = final.get("entry_price_low", original_strategy["entry_price_low"])
        original_strategy["entry_price_high"] = final.get("entry_price_high", original_strategy["entry_price_high"])
        original_strategy["stop_loss"] = final.get("stop_loss", original_strategy["stop_loss"])
        original_strategy["take_profit"] = final.get("take_profit", original_strategy["take_profit"])
        return original_strategy

    elif verdict == "降级执行":
        size_map = {"heavy": "medium", "medium": "light", "light": "light"}
        original_strategy["position_size"] = size_map.get(original_strategy.get("position_size", "light"), "light")
        return original_strategy

    elif verdict == "补充条件执行":
        original_strategy["execution_plan"] = final.get("execution_plan", original_strategy.get("execution_plan", ""))
        return original_strategy

    elif verdict == "推翻改为观望":
        _force_neutral(original_strategy, f"法官判决: {verdict}")
        return original_strategy

    return original_strategy
