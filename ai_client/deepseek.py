buimport os
import json
import time
import re
from datetime import datetime
from openai import OpenAI
from utils.logger import logger


# ==================== 配置参数 ====================
TICK_SIZE = 0.1
MAX_RETRIES = 2
RETRY_BASE_WAIT = 2
TIMEOUT_SECONDS = 180
# =================================================


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

    cross_context = ""
    if eth_data is not None:
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
清算池：上方{cross_above_liq:.2f}B / 下方{cross_below_liq:.2f}B（比值{cross_liq_ratio:.3f}）
情绪：OI分位{cross_oi_pct:.0f}%（24h{cross_oi_change:+.1f}%）、资金费率分位{cross_funding_pct:.0f}%、顶级多空比分位{cross_tls_pct:.0f}%（100%=机构极度看空）
资金流：CVD斜率{cross_cvd:.4f}（{cross_cvd_dir}）
期权：P/C比{cross_pc:.4f}、最大痛点{cross_max_pain:.2f}
"""

    prompt = f"""你是一个拥有十年经验管理200万U的顶尖加密货币短线交易员，精通清算动力学、多空博弈、技术分析、合约交易，必须根据以下指令进行深度分析，标准格式输出，不得简化或跳过，否则视为无效输出。
{constraint_note}
【{symbol} | {timestamp}】
价格：{current:.2f} | 15min ATR：{data['atr_15m']:.2f} | 1h ATR：{data.get('atr_1h', data['atr_15m']*2):.2f} | 波动因子：{data['vol_factor']:.2f} | 7日分位数：{data['price_percentile']:.0f}%

清算池：
上方(空头)：{data['above_liq']/1e9:.2f}B，{above_cluster} (距{above_distance})
下方(多头)：{data['below_liq']/1e9:.2f}B，{below_cluster} (距{below_distance})
比值：{data['liq_ratio']:.3f}

订单簿：买{data['orderbook_bids']/1e6:.1f}M / 卖{data['orderbook_asks']/1e6:.1f}M | 失衡率{data['orderbook_imbalance']:.4f}
资金费率：{data['funding_rate']:.4f}% (分位{data['funding_percentile']:.0f}%)
OI：{data['oi']/1e9:.2f}B (分位{data['oi_percentile']:.0f}%)，24h{data['oi_change_24h']:+.1f}%
全市场OI：{data['agg_oi']/1e9:.2f}B，24h{data['agg_oi_change_24h']:+.1f}%
顶级多空比：{data['top_ls_ratio']:.2f} (分位{data['top_ls_percentile']:.0f}%)
恐慌贪婪：{data['fear_greed']} (7日前{data['fear_greed_prev_7d']})
期权：最大痛点{max_pain:.2f} ({max_pain_bias}) | P/C比{put_call_ratio:.4f} ({pc_bias})
资金流：CVD斜率{data['cvd_slope']:.4f} | 期货24h净流{data['netflow']/1e6:.1f}M | 交易所BTC 24h变化{data['exchange_btc_change_24h']:+.0f} BTC
ETH/BTC：当前{eth_btc_ratio:.4f}，7日均值{eth_btc_ma_7d:.4f}，7日分位数{eth_btc_percentile:.0f}%（数值越高代表ETH相对BTC越强势）
数据缺失：{missing_str}
{cross_context}
---
【极其重要】你的最终回答中的 `reasoning` 字段必须是一个完整的、自包含的推演文本。它必须包含每一步的“分析数据”、“第一反应”、“自我质疑”、“最终结论”子标题及其详细内容。不得以摘要或简写形式输出。你的思考过程必须显式地写出来。
---
第一步：环境定调
分析数据：价格7日分位数、15min ATR、1h ATR、波动因子。
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
特别规则：若资金费率分位 > 80% 且 CVD斜率 > 0.1 且价格未跌破15min EMA12，则拥挤度信号仅作为止盈参考，不作为反转开仓依据。
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
分析数据：（逐项对比BTC与ETH的清算池规模/比值、OI分位及24h变化、资金费率分位、顶级多空比分位、CVD斜率、P/C比、ETH/BTC汇率分位，分析各币种的方向，是一致还是相反，深度分析矛盾点或共振点）
【重要：顶级多空比分位的正确解读】顶级多空比分位100%意味着当前顶级交易员的净多头持仓比例处于历史最低水平，即他们正以历史极端比例持有净空头。这是强烈的看跌信号，而非轧空信号。扎空逻辑成立的前提是“散户极度做空 + 资金流强劲”，而顶级多空比反映的是机构行为，两者不可混淆。
第一反应：
自我质疑：
最终结论：

第七步：矛盾裁决与决策
交叉验证与裁决：将前六步的所有最终结论逐条列出，明确标注每条是“看涨”还是“看跌”。然后基于你作为顶尖交易员的经验，独立、自主地分配各步证据的权重，并明确写出你为何如此分配权重。
【关键要求】你必须在裁决中回答以下问题：
1. 在当前市场环境下，哪一个信号是我认为最不可忽视的？为什么？
2. 与我方向相反的最强信号是什么？我为什么决定压倒它（或者被它压倒）？
3. 我的最终方向是建立在哪一两个核心假设之上的？如果这些假设被证伪，我会立刻改变方向。

方向选择（long/short/neutral）：
置信度（high/medium/low）：
仓位（light/medium/heavy）：

入场区间（说明依据）：
止损位（说明依据）：
止盈位（说明依据）：
主动证伪信号：
微观盘口确认：

输出JSON（不要代码块）：
{{
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
        base_url="https://api.deepseek.com/v1",
        timeout=TIMEOUT_SECONDS
    )
    for attempt in range(max_retries):
        try:
            logger.info(f"DeepSeek 调用 (尝试 {attempt+1}/{max_retries})")
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=6000,
                timeout=TIMEOUT_SECONDS
            )
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

    # 计算实际盈亏比并写入字典，供显示使用
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

    return True, ""
