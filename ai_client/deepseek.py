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

def build_prompt(data: dict, symbol: str, eth_data: dict = None) -> str:
    timestamp = data.get("timestamp", "N/A")
    current = data['mark_price']
    above_cluster = data.get('above_cluster', 'N/A')
    below_cluster = data.get('below_cluster', 'N/A')

    above_distance = "N/A"
    below_distance = "N/A"
    if above_cluster != 'N/A' and '-' in above_cluster:
        parts = above_cluster.split('-')
        above_high = float(parts[1])
        above_distance = f"+{above_high - current:.0f}"
    if below_cluster != 'N/A' and '-' in below_cluster:
        parts = below_cluster.split('-')
        below_low = float(parts[0])
        below_distance = f"-{current - below_low:.0f}"

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
        if not eth_data.get("_complete", False):
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
            cross_pc = eth_data.get('put_call_ratio', 0)
            cross_max_pain = eth_data.get('max_pain', 0)

            cross_cvd_dir = "正（买盘）" if cross_cvd > 0 else "负（卖盘）"

            cross_context = f"""
【跨币种辅助验证数据】
现价：{cross_current:.2f} | ETH/BTC汇率7日分位：{eth_btc_percentile:.0f}%
清算池：上方{cross_above_liq:.2f}B(约{cross_above_liq*10:.0f}亿美元) / 下方{cross_below_liq:.2f}B(约{cross_below_liq*10:.0f}亿美元)（比值{cross_liq_ratio:.3f}）
情绪：OI分位{cross_oi_pct:.0f}%（24h{cross_oi_change:+.1f}%）、资金费率分位{cross_funding_pct:.0f}%、顶级多空比分位{cross_tls_pct:.0f}%（100%=机构极度看空）
资金流：CVD斜率{cross_cvd:.4f}（{cross_cvd_dir}）
期权：P/C比{cross_pc:.4f}、最大痛点{cross_max_pain:.2f}
"""

    prompt = f"""你是一个拥有十年经验管理200万U的顶尖加密货币短线交易员，精通清算动力学、多空博弈、技术分析、合约交易。
{constraint_note}
【{symbol} | {timestamp}】
价格：{current:.2f} | 15min ATR：{data['atr_15m']:.2f} | 1h ATR：{data.get('atr_1h', data['atr_15m']*2):.2f} | 1h波动率：{data.get('atr_1h_ratio', 0):.2f}% | 波动因子：{data['vol_factor']:.2f} | 7日分位数：{data['price_percentile']:.0f}%

清算池：
上方(空头)：{data['above_liq']/1e9:.2f}B (约{data['above_liq']/1e7:.0f}亿美元)，{above_cluster} (距{above_distance})
下方(多头)：{data['below_liq']/1e9:.2f}B (约{data['below_liq']/1e7:.0f}亿美元)，{below_cluster} (距{below_distance})
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
必须且只能引用上方提供的具体数据，不得编造、估算或使用记忆中的任何数值。你的思考过程必须显式地写出来，不得简化或跳过。
必须输出纯文本格式，不得添加任何表情符号或特殊字符，不得以摘要或简写形式输出。
---
第一步：环境定调
分析数据：价格7日分位数、1h波动率、波动因子。
第一反应：
自我质疑：
最终结论：

第二步：猎物定位
分析数据：上下方清算池距离/强度、比值、订单簿买卖盘量、失衡率。
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

第六步：跨币种验证
分析数据：（必须完成以下三项强制对比，每项写出具体数值）
① 清算池比值方向对比：BTC比值 [ ]，ETH比值 [ ]，两者方向是否一致？
② CVD斜率方向对比：BTC CVD = [ ]，ETH CVD = [ ]，资金流向是否共振？
③ 顶级多空比分位对比：BTC分位 [ ]%，ETH分位 [ ]%，哪个币种的机构空头更极端？
基于此三组客观对比，深度分析两币种整体方向是一致还是矛盾，并解释共振或背离的市场含义。
第一反应：
自我质疑：
最终结论：

第七步：矛盾裁决与决策
请基于前六步的所有数据和结论，独立做出最终交易决策。

你需要：
1. 指出前六步中最强的看涨信号和最强的看跌信号。
2. 解释你如何权衡这些矛盾信号——为什么你选择相信某些信号，而压低另一些信号的权重。
3. 明确写出你的核心假设——你判断方向的核心逻辑链是什么。
4. 给出证伪条件——必须包含一个具体的、基于价格行为的证伪条件（例如：价格以1小时K线收盘价站上/跌破某个具体点位），一旦触发，你的核心假设即被推翻，你必须立即暂停原计划。
5. 输出最终方向、置信度、仓位。

【短线交易原则】
你的交易方向应与价格推演中最先发生的显著运动同向。若你的方向与第一段运动相反，你必须选择观望或顺应第一段运动的方向。

随后完成流动性猎杀推演专业研判（价格路径推演、触发条件、证伪标准）以及入场区间、止损位、止盈位、主动证伪信号、微观盘口确认等。
---
推理自检：
1. 我的最终裁决是否完全基于前六步的数据和结论？
2. 我在哪一步的“自我质疑”中发现了后来被证实为关键的风险点？
3. 如果我错了，最可能是在哪一步的假设上栽了跟头？
4. 我是否遵守了【短线交易原则】？若方向与第一段运动相反，我是否已改为观望或顺势？

入场区间（说明依据）：
止损位（说明依据）：
止盈位（说明依据）：
主动证伪信号：
微观盘口确认：
{{
  "decision_summary": {{
    "final_direction": "看涨/看跌/观望",
    "first_leg_direction": "上涨/下跌",
    "first_leg_magnitude": 0.0,
    "effective_threshold": 0.0,
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
  "reasoning": "完整的七步推演内容，必须包含价格路径推演与推理自检",
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
    s["direction"] = "neutral"
    s["confidence"] = "low"
    s["position_size"] = "none"
    s["entry_price_low"] = 0
    s["entry_price_high"] = 0
    s["stop_loss"] = 0
    s["take_profit"] = 0
    s["execution_plan"] = ""
    s["reasoning"] += f"\n\n[系统自动干预] 原始信号因违反短线铁律被强制改为观望。原因：{reason}"
    s["risk_note"] = f"[系统干预] 原始方向因违反短线铁律被强制改为观望。原始风险说明：{s.get('risk_note', '')}"

def validate_strategy(s: dict, data: dict = None) -> tuple[bool, str]:
    direction = s.get("direction")
    if direction not in ["long", "short", "neutral"]:
        return False, f"无效方向: {direction}"

    if direction == "neutral":
        for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
            if s.get(f, 0) != 0:
                return False, f"neutral 信号不应有非零的 {f}"
        return True, ""

    for f in ["entry_price_low", "entry_price_high", "stop_loss", "take_profit"]:
        val = s.get(f)
        if val is None or float(val) <= 0:
            return False, f"缺少或无效的 {f}"

    if float(s["entry_price_low"]) > float(s["entry_price_high"]):
        return False, "入场区间下限大于上限"

    try:
        entry_low = float(s["entry_price_low"])
        entry_high = float(s["entry_price_high"])
        stop = float(s["stop_loss"])
        tp = float(s["take_profit"])

        if direction == "long":
            worst_entry = entry_high
            risk = worst_entry - stop
            reward = tp - worst_entry
        else:
            worst_entry = entry_low
            risk = stop - worst_entry
            reward = worst_entry - tp

        if risk > 0:
            s["_calculated_rr"] = round(reward / risk, 2)
        else:
            s["_calculated_rr"] = 0.0
    except Exception as e:
        logger.warning(f"计算盈亏比时出错: {e}")
        s["_calculated_rr"] = None

    # decision_summary 校验
    summary = s.get("decision_summary", {})
    if summary:
        final_dir = summary.get("final_direction", "")
        first_leg = summary.get("first_leg_direction", "")

        if final_dir == "看涨" and direction != "long":
            _force_neutral(s, f"decision_summary最终方向为看涨，但direction为{direction}")
            return True, "已自动修正为观望"
        if final_dir == "看跌" and direction != "short":
            _force_neutral(s, f"decision_summary最终方向为看跌，但direction为{direction}")
            return True, "已自动修正为观望"
        if final_dir == "观望" and direction != "neutral":
            _force_neutral(s, f"decision_summary最终方向为观望，但direction为{direction}")
            return True, "已自动修正为观望"

        if first_leg == "下跌" and direction == "long":
            _force_neutral(s, "第一段运动为下跌，但输出做多，违反短线铁律")
            return True, "已自动修正为观望"
        if first_leg == "上涨" and direction == "short":
            _force_neutral(s, "第一段运动为上涨，但输出做空，违反短线铁律")
            return True, "已自动修正为观望"

    # 原有的一致性校验（推演路径与方向矛盾）
    reasoning = s.get("reasoning", "")
    atr_1h = data.get("atr_1h", data.get("atr_15m", 0) * 2) if data else 0

    if direction in ("long", "short") and atr_1h > 0:
        first_leg_down = re.search(r'(?:先.*?下[跌挫].*?再.*?上[涨升])|(?:第一段.*?下跌)', reasoning)
        first_leg_up = re.search(r'(?:先.*?上[涨升].*?再.*?下[跌挫])|(?:第一段.*?上涨)', reasoning)

        if first_leg_down and direction == "long":
            _force_neutral(s, "推演路径先跌后涨，但输出做多，违反短线铁律")
            return True, "已自动修正为观望"
        if first_leg_up and direction == "short":
            _force_neutral(s, "推演路径先涨后跌，但输出做空，违反短线铁律")
            return True, "已自动修正为观望"

        if "不应做多" in reasoning and direction == "long":
            _force_neutral(s, "推理明确声明不应做多，但输出long")
            return True, "已自动修正为观望"
        if "不应做空" in reasoning and direction == "short":
            _force_neutral(s, "推理明确声明不应做空，但输出short")
            return True, "已自动修正为观望"

    final_decision_match = re.search(r'最终方向[：:]\s*(看涨|看跌|观望)', reasoning)
    if final_decision_match:
        decision_text = final_decision_match.group(1)
        inferred = {"看涨": "long", "做多": "long", "看跌": "short", "做空": "short", "观望": "neutral"}.get(decision_text)
        if inferred and inferred != direction:
            _force_neutral(s, f"推理最终方向为{decision_text}，但JSON输出为{direction}")
            return True, "已自动修正为观望"

    return True, ""