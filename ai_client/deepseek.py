import os, json, time, re
from datetime import datetime
from openai import OpenAI
from utils.logger import logger

# 配置
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

    max_pain, put_call_ratio = data['max_pain'], data['put_call_ratio']
    max_pain_bias = "偏空信号" if current > max_pain else "偏多信号"
    pc_bias = "偏空信号" if put_call_ratio > 1.0 else "偏多信号"

    eth_btc_ratio = data['eth_btc_ratio']
    eth_btc_ma_7d = data.get('eth_btc_ma_7d', 0.0)
    eth_btc_percentile = data.get('eth_btc_percentile', 50.0)

    core_missing = [k for k in ["atr_15m", "above_liq", "below_liq", "cvd_slope"] if k in missing]
    constraint_note = ""
    if core_missing:
        constraint_note = f"\n【重要约束】核心数据缺失：{', '.join(core_missing)}，置信度设为 'low'；若清算数据缺失，输出 'neutral'。\n"

    # 构建 ETH 辅助数据块
    eth_context = ""
    if eth_data is not None:
        eth_c = eth_data.get('mark_price', 0)
        eth_context = f"""
【ETH 辅助验证数据】
ETH现价：{eth_c:.2f} | ETH/BTC汇率7日分位：{eth_btc_percentile:.0f}%
ETH清算池：上方{eth_data['above_liq']/1e9:.2f}B / 下方{eth_data['below_liq']/1e9:.2f}B（比值{eth_data['liq_ratio']:.3f}）
ETH情绪：OI分位{eth_data['oi_percentile']:.0f}%（24h{eth_data['oi_change_24h']:+.1f}%）、资金费率分位{eth_data['funding_percentile']:.0f}%、顶级多空比分位{eth_data['top_ls_percentile']:.0f}%
ETH资金流：CVD斜率{eth_data['cvd_slope']:.4f}（{'正（买盘）' if eth_data['cvd_slope']>0 else '负（卖盘）'}）
ETH期权：P/C比{eth_data['put_call_ratio']:.4f}、最大痛点{eth_data['max_pain']:.2f}
"""

    prompt = f"""你是拥有十年经验管理200万U的顶尖加密货币短线交易员，精通清算动力学、多空博弈、技术分析。你必须严格遵循七步推演框架，每一步进行真正的深度思考，不得走形式或跳过推理环节。

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
ETH/BTC：当前{eth_btc_ratio:.4f}，7日均值{eth_btc_ma_7d:.4f}，7日分位数{eth_btc_percentile:.0f}%
数据缺失：{missing_str}
{eth_context}
---
【七步推演框架】
每一步必须包含“分析数据”、“第一反应”、“自我质疑”、“最终结论”四个子标题，且必须严格符合深度思考要求。

第一步：环境定调
分析数据：价格7日分位数、15min ATR、1h ATR、波动因子。（必须写出具体数值）
第一反应：基于数据给出初步方向性判断，并说明逻辑链。
自我质疑：至少提出一个实质性反证，与当前数据挂钩。
最终结论：明确市场状态和策略基调，写出成立前提。

第二步：猎物定位
分析数据：上下方清算池距离/强度、比值、订单簿买卖盘量、失衡率。（写出具体数值）
第一反应：判断大资金最可能猎杀方向。
自我质疑：至少一个实质性反证。
最终结论：明确猎物方向，写出成立前提。
（特别规则：若清算比值在0.8-1.2且距离差<20%，则结论必须为“清算结构对称，方向不明确”）

第三步：对手盘解剖
分析数据：OI分位及24h变化、全市场OI变化、资金费率分位、顶级多空比分位、恐慌贪婪及趋势。（写出具体数值）
第一反应：判断市场拥挤度和脆弱方。
自我质疑：至少一个实质性反证。
最终结论：明确谁将成为燃料，写出成立前提。
（特别规则：若资金费率分位>80%且CVD斜率>0.1且价格未跌破15min EMA12，则拥挤度信号仅用于止盈参考，不做反转开仓依据）

第四步：资金流验证
分析数据：CVD斜率方向/量级、期货24h净流、交易所BTC余额变化。（写出具体数值）
第一反应：判断资金流是否支持猎物方向。
自我质疑：至少一个实质性反证。
最终结论：明确共振或背离，写出成立前提。
（特别规则：若三个指标中有两个以上方向不一致，则结论必须输出“资金流信号矛盾，方向不明确”）

第五步：辅助信号扫描
分析数据：期权最大痛点、P/C比、ETH/BTC汇率。（写出具体数值）
第一反应：判断信号加强还是削弱主逻辑。
自我质疑：至少一个实质性反证。
最终结论：明确净影响，写出成立前提。

第六步：跨币种验证
分析数据：逐项对比BTC与ETH的以下指标（每项写出两币种具体数值和差异方向）：清算池规模/比值、OI分位及24h变化、资金费率分位、顶级多空比分位、CVD斜率、P/C比、ETH/BTC汇率分位。
第一反应：判断两币种信号是共振还是矛盾。
自我质疑：ETH数据是否因币种特性而失真？
最终结论：明确对BTC主逻辑的增强/削弱程度，写出该结论的特定前提。
（若ETH数据不可用，结论必须输出“ETH辅助数据不可用，跨币种验证无法进行，对主逻辑无增强也无削弱”）

第七步：矛盾裁决与决策
交叉验证与裁决：逐条列出前六步的印证点与矛盾点。基于交易经验，自主分配各步证据权重，并明确写出权重分配逻辑。若矛盾严重无法化解，输出neutral。
价格路径推演（流动性猎杀推演）：必须专业研判，基于当前清算池分布、对手盘结构和资金流方向，描述价格最可能如何测试并触发关键流动性区域，以及触发后可能产生的连锁反应。需包含触发条件和证伪标准。
推理自检：
1. 我的最终裁决是否完全基于前六步的数据和结论？
2. 我在哪一步的“自我质疑”中发现了后来被证实为关键的风险点？
3. 如果我错了，最可能是在哪一步的假设上栽了跟头？

随后输出：方向选择（long/short/neutral）、置信度（high/medium/low）、仓位（light/medium/heavy）、入场区间及依据、止损位及依据、止盈位及依据、主动证伪信号、微观盘口确认。

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
    if m: return m.group(1).strip()
    m = re.search(r'```\s*([\s\S]*?)\s*```', content)
    if m: return m.group(1).strip()
    start = content.find('{')
    if start == -1: raise ValueError("未找到 JSON")
    count = 0
    for i, c in enumerate(content[start:], start):
        if c == '{': count += 1
        elif c == '}':
            count -= 1
            if count == 0: return content[start:i+1].strip()
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
                model="deepseek-chat",  # V4模型
                messages=[{"role": "user", "content": prompt}],
                max_tokens=6000,
                timeout=TIMEOUT_SECONDS
            )
            content = resp.choices[0].message.content or ""
            reasoning = getattr(resp.choices[0].message, 'reasoning_content', None)
            _log_response(prompt, content, reasoning)

            final_content = content.strip() if content else (reasoning or "")
            if not final_content: raise ValueError("响应内容为空")
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
                time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
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
        for f in ["entry_price_low","entry_price_high","stop_loss","take_profit"]:
            if s.get(f, 0) != 0: return False, f"neutral 信号不应有非零的 {f}"
        return True, ""
    for f in ["entry_price_low","entry_price_high","stop_loss","take_profit"]:
        val = s.get(f)
        if val is None or float(val) <= 0: return False, f"缺少或无效的 {f}"
    if float(s["entry_price_low"]) > float(s["entry_price_high"]):
        return False, "入场区间下限大于上限"
    return True, ""
