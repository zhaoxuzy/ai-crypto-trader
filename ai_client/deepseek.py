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

    # 自动推断对比币种
    if cross_symbol is None:
        if symbol == "BTC":
            cross_symbol = "ETH"
        elif symbol == "ETH":
            cross_symbol = "BTC"
        elif symbol == "SOL":
            cross_symbol = "BTC"
        else:
            cross_symbol = "ETH"

    # ===== 触发距与远边界距 =====
    above_trigger = "N/A"          # 空头池：到最近边（下沿）的距离
    above_far_boundary = "N/A"     # 空头池：到最远边（上沿）的距离
    below_trigger = "N/A"          # 多头池：到最近边（上沿）的距离
    below_far_boundary = "N/A"     # 多头池：到最远边（下沿）的距离

    if above_cluster != 'N/A' and '-' in above_cluster:
        parts = above_cluster.split('-')
        above_low = float(parts[0])   # 下沿（近边）
        above_high = float(parts[1])  # 上沿（远边）
        above_trigger = f"+{above_low - current:.0f}"
        above_far_boundary = f"+{above_high - current:.0f}"
    if below_cluster != 'N/A' and '-' in below_cluster:
        parts = below_cluster.split('-')
        below_low = float(parts[0])   # 下沿（远边）
        below_high = float(parts[1])  # 上沿（近边）
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

    # ===== 跨币种辅助数据 =====
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

    # 动态生成第六步对比文本
    step6_text = f"""第六步：跨币种验证
分析数据：（必须完成以下三项强制对比，每项写出具体数值）
① 清算池比值方向对比：{symbol}比值 [ ]，{cross_symbol}比值 [ ]，两者方向是否一致？
② CVD斜率方向对比：{symbol} CVD = [ ]，{cross_symbol} CVD = [ ]，资金流向是否共振？
③ 顶级多空比分位对比：{symbol}分位 [ ]%，{cross_symbol}分位 [ ]%，哪个币种的机构空头更极端？
【基于此三组客观对比，深度分析两币种整体方向是一致还是矛盾，并解释共振或背离的市场含义】
第一反应：
自我质疑：
最终结论：
【跨币种反馈】基于以上对比，我的（{symbol}/{cross_symbol}）单币种结论被如何修正？有无新增风险点？如果有，我必须回到第三步或第五步的自我质疑中重新审视。
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
{cross_context}
---
【硬性约束】
1. 必须且只能引用上方提供的具体数据，不得编造、估算或使用记忆中的任何数值。
2. 你的 `reasoning` 字段必须包含从第一步到第九步的完整推演文本，每一步都必须显式包含“分析数据”、“第一反应”、“自我质疑”、“最终结论”子标题及详细内容。

---
第一步：环境定调
分析数据：价格7日分位数、1h波动率、波动因子。
第一反应：
自我质疑：
最终结论：

第二步：猎物定位
分析数据：上下方清算池距离/强度、比值、订单簿买卖盘量、失衡率。
【猎物定位规则】“猎物距离”必须使用触发距，而非远边界距。价格触碰清算区上沿（多头池）或下沿（空头池）即可开始触发清算。你在预估第一段运动幅度时，必须以触发距为基准进行浮动估计。
【幅度预估规则】你必须以下列三步推演来估算第一段运动的潜在幅度，不得跳过任何一步：
1基准触发距：[ ]点。这是清算引擎启动的最小距离。
2惯性穿透力评估：当前抛压/买盘是否足以让价格击穿清算区边缘后继续深入？
   - 请依据 订单簿失衡率 和 CVD斜率真实强度 回答：“强”、“中等”或“弱”。
   - 若为“强”，价格大概率会穿透至清算区内部，你的预估幅度应覆盖更深的区域；
   - 若为“弱”，价格可能仅触发边缘后便停滞或反弹，你的预估幅度应较浅。
3幅度预估：基于以上推演，你预估的第一段运动幅度为 [ ] 点。简述这个数值的战术依据。
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
第一反应：
自我质疑：
最终结论：

第五步：辅助信号扫描
分析数据：期权最大痛点、P/C比、ETH/BTC汇率。
第一反应：
自我质疑：
最终结论：

{step6_text}

第七步：交叉验证与裁决【你刚刚完成了对市场所有维度的扫描。现在，你面对的可能是一个多空交织的矛盾体。请基于前六步的所有数据和结论，独立完成以下裁决。你的回答必须明确包含最终方向、置信度和仓位】
信号传唤：
   - 明确指出前六步中，哪一个信号是你见过最强的看涨证据，它来自哪一步？
   - 明确指出前六步中，哪一个信号是你见过最强的看跌证据，它来自哪一步？
权重审判：
   - 为你做最终裁决所依据的关键决策因子分配权重（合计100%）。
   - 必须简要说明：在当前的市场阶段（是处于趋势的早期、末期，还是震荡中），为何你会赋予某些因子压倒性的权重，而让另一些因子近乎失效？
心证交锋：
   - 你必须模拟你的内心对话：扮演一个死硬空头（或根据最强看跌信号的立场）与你辩论，用最强看涨证据攻击你的空头逻辑。
   - 然后，你再以主交易员的身份，驳斥或接纳这个攻击。
   - 最终解释，你为何在当前的特定时间、特定价位下，选择压过最强反向信号的干扰，坚持你的方向判断。若你被反向证据说服，你必须立即改变立场，选择观望。
核心假设：
   - 清晰写出你判断方向的核心逻辑链条。它不是事实的堆砌，而是你脑海中的那份战术蓝图：“因为[关键假设A]，所以会出现[现象B]，进而引发[C的行情结果]。”
证伪条件：
   - 设定你的“逃生门”：具体的价格行为（如1H收盘价站上/跌破X），以及必须同时发生的资金流或订单簿确认。
   - 声明：仅有价格穿刺而无资金流确认的突破，将被我视为陷阱。两者必须共同触发，我的核心假设才被推翻，我将无条件执行撤离。

第八步：交易执行计划【基于第七步的最后裁决结果，制定可直接执行的具体计划】
1.价格路径推演：
   - 利用流动性猎杀进行价格推演，描述当前价格的走势，最可能如何测试并触发关键流动性区域，以及触发后可能产生的连锁反应，说明触发条件是说明，什么现象出现，什么情况价格路径推演将失效。
2.合约策略：
   -入场区间（说明依据）
   -止损位（说明依据）
   -止盈位（说明依据）
3.主动证伪信号：
4.微观盘口确认：

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

    # 核心数据缺失强制拦截
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

    # neutral 信号校验
    if direction == "neutral":
        for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
            if s.get(f, 0) != 0:
                return False, f"neutral 信号不应有非零的 {f}"
        return True, ""

    # 价格字段有效性
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

    # 止损位几何合理性检查（仅提示，不拦截）
    if direction == "long" and stop_loss >= entry_low:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间下方，请人工确认。"
    elif direction == "short" and stop_loss <= entry_high:
        s["risk_note"] = s.get("risk_note", "") + " [系统提示] 止损位未处于入场区间上方，请人工确认。"

    # 计算盈亏比
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

    # decision_summary 与 direction 的一致性校验（强制转为 neutral）
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
