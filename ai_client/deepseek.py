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
2. 你的 `reasoning` 字段必须包含从第一步到第六步的完整推演文本，每一步都必须显式包含“分析数据”、“第一反应”、“自我质疑”、“最终结论”子标题及详细完整内容。
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
分析数据：完成以下三项强制对比，每项写出具体数值，基于此三组客观对比，深度分析两币种整体方向是一致还是矛盾，并解释共振或背离的市场含义.
① 清算池比值方向对比：[ ] vs [ ]，两者方向是否一致？
② CVD斜率方向对比：[ ] vs [ ]，资金流向是否共振？
③ 顶级多空比分位对比：[ ]% vs [ ]%，哪个币种的机构空头更极端？
第一反应：
自我质疑：
最终结论：
跨币种反馈：基于以上对比，我的单币种结论被如何修正？有无新增风险点？如果有，我必须回到第三步或第五步的自我质疑中重新审视。

第七步：交易计划【基于前六步的分析，你直接做出方向判断并制定具体的交易计划。你的责任是交易员，不需要在此处自我审查或裁决。】
方向判断：我决定做多 / 做空 / 观望。我的置信度为 high / medium / low，仓位为 heavy / medium / light。
价格路径推演：必须综合运用流动性猎杀理论（清算池位置/强度）、行为金融学（对手盘心理、恐慌/贪婪、顶级多空比极端值）以及博弈论（做市商与散户的短期博弈策略）进行推演，价格最可能如何测试并触发关键流动性区域？触发后会产生何种连锁强平或踩踏？
合约策略：入场区间+止损位+止盈位，说明具体的依据。
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
  "reasoning": "第一步到第七步的完整推演文本，包含所有分析、第一反应、自我质疑、最终结论以及价格路径推演",
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

    return f"""你是我们加密货币交易团队的**风控审计官**。你的唯一任务是精准、快速找出首席交易员策略中的错误。你只找错误，不做裁决，不写建议。

【交易标的】{symbol}
【代码层客观锚点】清算池综合吸引力评分：{bias_text}（基于规模/触发距/订单簿计算，数学模型得出）
【原策略方向】{original_strategy.get('direction')}
【入场/止损/止盈】{original_strategy.get('entry_price_low')} - {original_strategy.get('entry_price_high')} / {original_strategy.get('stop_loss')} / {original_strategy.get('take_profit')}

【原策略推演过程】
{original_strategy.get('reasoning', '无推演过程')}

【审查要求】
请严格按照以下模板输出，只输出报告内容，不要额外解释：

【风控审计官 - 审计报告】

一、数据与解读错误
- [若有错误，按此格式：在[步骤X]中，A声称[数值/解读]，但实际数据为[数值/正确含义]。此错误[是否影响方向判断]。 [严重性：高/中/低]]
- [若无，写“未发现数据或解读错误”]

二、逻辑错误
- [若有错误，按此格式：[错误类型]在[步骤X]：[描述]。 [严重性：高/中/低]]
- [若无，写“未发现明显逻辑错误”]

三、关键反证提示
- [若选择性忽略了某个关键反向数据，按此格式：在[步骤X]中，[反向数据X]被忽略或低估，该数据暗示[方向]，可能与策略结论构成矛盾。 [严重性：高/中/低]]

四、博弈层面审视
- 策略假设做市商站在哪一边？做市商是否有可能反向利用预设的止损簇来猎杀流动性？
- 预设的入场点位是否恰好是另一类聪明钱（如趋势跟踪基金、期权做市商）的理想反向入场点？
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
                max_tokens=4096,
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
    market_data_str = json.dumps(data, ensure_ascii=False)

    prompt = f"""你是最终决策的交易委员会主席，拥有二十年加密货币短线合约交易经验。你的团队已准备了【首席交易员的策略】和【风控审计报告】。你必须基于事实与逻辑，输出一份可立即执行的短信交易方案。

【交易标的】{symbol}
【原策略】方向：{orig_dir}，入场：{entry_l}-{entry_h}，止损：{stop_l}，止盈：{tp_l}
【推演过程】{original_strategy.get('reasoning', '无')}
【审计报告】{report}
【市场数据】{market_data_str}

【裁决流程】
你必须严格遵循以下两个步骤进行，不可跳过或自由发挥。
STEP 1 – 审计逐条裁决
对审计报告中的每一条指控，你必须进行如下操作：
  - 给出你的：裁决结论（采纳 / 驳回 / 部分采纳）
  - 陈述数据：核验依据（必须引用上方“市场数据”中的字段名和具体数值或专业的判断）
  - 进行：反证风险评估（高/中/低）。你必须提出至少一条可能挑战你裁决的证据。反证必须包含：该证据是什么、它为何可能构成威胁、以及你为何最终排除它。若你搜寻了所有数据后确实认为毫无矛盾，才可填写“无”，但必须简述你的搜寻范围作为理由。

STEP 2 – 制定最终策略
  (A) 判断你是否维持原策略的方向。
  (B) 如果你决定**推翻原方向**，你必须同时完成以下四项工作，缺一不可：
      1) 明确指出原方向所依赖的一个核心假设（一句话概括即可，例如：“假设市场将因卖盘衰竭而反弹”）。
      2) 引用市场数据中的具体数值，证明该核心假设已被证伪。
      3) 给出支持新方向的两个独立数据指标，每个都必须附带字段名和当前数值。
      4) 给出新策略的入场区间、止损位、止盈位，并简要说明为何如此设置（例如：基于某个关键结构或流动性区域）。
STEP 3 – 按照标准模板输出最终策略
        -输出格式要求：纯文本，不要输出JSON或其他任何格式。
        -输出模板：请严格按照以下模板输出。

📌 最终判决：[维持原判 / 修正参数 / 降级执行 / 推翻]
🎯 执行指令：
   - 方向：[做多 / 做空 / 观望]
   - 仓位：[轻仓 / 中仓 / 重仓 / 无]
   - 入场区间：[价格下限-价格上限]（说明依据，若观望则写“无”）
   - 止损：[价格]（说明依据，若观望则写“无”）
   - 止盈：[价格]（说明依据，若观望则写“无”）
   - 说明：[一句话指令，或观望时的触发条件]
📋 裁决说明：
   -在此处逐条列出对审计指控的裁决，每条必须包含：指控内容、裁决(采纳/驳回)、依据(说明依据，引用具体数据)
   例如：1.指控内容：[步骤二] 将30.39B解读为3000亿，数量级夸大。
           裁决结论：采纳/驳回/部分采纳
           核验依据：xxx
           反证风险评估：xx
   -核心逻辑：简述执行指令中策略制定逻辑，必须提供站的住脚的证据或推论。
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
                max_tokens=4096,
                timeout=120
            )
            content = resp.choices[0].message.content or ""
            _log_response(prompt, content)
            if not content.strip():
                raise ValueError("交易委员会响应为空")

            # ---------- 纯文本解析 ----------
            original_dir = original_strategy.get("direction", "neutral")
            direction = original_dir
            position_size = original_strategy.get("position_size", "none")
            entry_low = original_strategy.get("entry_price_low", 0)
            entry_high = original_strategy.get("entry_price_high", 0)
            stop_loss = original_strategy.get("stop_loss", 0)
            take_profit = original_strategy.get("take_profit", 0)
            execution_plan = ""

            verdict_match = re.search(r'📌\s*最终判决[：:]\s*(.*)', content)
            title_line = verdict_match.group(0).strip() if verdict_match else "📌 最终判决：维持原判"

            exec_section = re.search(r'🎯\s*执行指令[：:]?\s*(.*?)(?=📋|⚠️|$)', content, re.DOTALL)
            if exec_section:
                exec_text = exec_section.group(1).strip()
                # 1. 方向解析（兼容中英文）
                dir_match = re.search(r'方向[：:]\s*(long|short|neutral)', exec_text)
                if dir_match:
                    direction = dir_match.group(1)
                else:
                    # 尝试匹配中文
                    if re.search(r'方向[：:]\s*做多', exec_text):
                        direction = "long"
                    elif re.search(r'方向[：:]\s*做空', exec_text):
                        direction = "short"
                    elif re.search(r'方向[：:]\s*(观望|中性)', exec_text):
                        direction = "neutral"
                
                # 2. 仓位
                pos_match = re.search(r'仓位[：:]\s*(light|medium|heavy|none)', exec_text)
                if pos_match:
                    position_size = pos_match.group(1)
                # 3. 入场区间
                entry_match = re.search(r'入场区间[：:]\s*([\d.]+)\s*[-–]\s*([\d.]+)', exec_text)
                if entry_match:
                    entry_low = float(entry_match.group(1))
                    entry_high = float(entry_match.group(2))
                # 4. 止损
                stop_match = re.search(r'止损[：:]\s*([\d.]+)', exec_text)
                if stop_match:
                    stop_loss = float(stop_match.group(1))
                # 5. 止盈
                tp_match = re.search(r'止盈[：:]\s*([\d.]+)', exec_text)
                if tp_match:
                    take_profit = float(tp_match.group(1))
                # 6. 说明
                plan_match = re.search(r'说明[：:]\s*(.*)', exec_text)
                if plan_match:
                    execution_plan = plan_match.group(1).strip()
                else:
                    execution_plan = exec_text.replace('\n', ' ').strip()
            else:
                exec_text = ""

            # 简化判决：以 direction 为准
            if direction == "neutral":
                verdict = "推翻改为观望"
                entry_low, entry_high, stop_loss, take_profit = 0, 0, 0, 0
                position_size = "none"
            elif direction != original_dir:
                verdict = "推翻改为反向操作"
            else:
                if verdict_match:
                    verdict_text = verdict_match.group(1).strip()
                    if "修正" in verdict_text:
                        verdict = "修正参数"
                    elif "降级" in verdict_text:
                        verdict = "降级执行"
                    else:
                        verdict = "维持原判"
                else:
                    verdict = "维持原判"

            reasoning_block = ""
            reason_section = re.search(r'📋\s*裁决理由[：:]?\s*(.*?)(?=⚠️|$)', content, re.DOTALL)
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
                    "title_line": title_line,
                    "exec_block": exec_section.group(0).strip() if exec_section else "",
                    "reasoning_block": reasoning_block,
                    "risk_block": risk_block
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
    # 避免因文本中提到“放弃做空，转向做多”而误判方向矛盾
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
        return original_strategy

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
        _validate_execution_direction(original_strategy)
        return original_strategy

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
        _validate_execution_direction(original_strategy)
        return original_strategy

    elif verdict == "推翻改为观望":
        _force_neutral(original_strategy, f"交易委员会决议: {verdict}")
        original_strategy["_judge_reasoning"] = _clean_reasoning(final.get("reasoning", ""))
        return original_strategy

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
        _validate_execution_direction(original_strategy)
        return original_strategy

    return original_strategy