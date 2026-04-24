import os
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

    # === 跨币种数据完整性检查 ===
    cross_context = ""
    if eth_data is not None:
        crucial_fields = ['above_liq', 'below_liq', 'oi_percentile', 'funding_percentile', 'top_ls_percentile', 'cvd_slope', 'put_call_ratio', 'max_pain']
        if all(eth_data.get(field, 0) == 0 for field in crucial_fields):
            cross_context = "\n【重要：跨币种数据不完整，第六步跨币种验证无法进行，对主逻辑无增强也无削弱。第七步宏裁决必须跳过跨币种对比。】\n"
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
必须且只能引用上方提供的具体数据，不得编造、估算或使用记忆中的任何数值。所有分析必须基于给定数据，否则输出无效。
最终回答中的 `reasoning` 字段必须是一个完整的、自包含的推演文本，它必须包含每一步的“分析数据”、“第一反应”、“自我质疑”、“最终结论”子标题及其详细内容，你的思考过程必须显式地写出来，不得简化或跳过。
必须输出纯文本格式，不得添加任何表情符号或特殊字符，不得以摘要或简写形式输出。
必须根据以下七步指令以顶尖加密货币交易员角色进行深度分析，犯基本的数据指标解读错误是完全不能接受的，
---
第一步：环境定调
分析数据：价格7日分位数、1h波动率、波动因子。
【波动率参照】1h波动率<0.2%为低波，0.2%-0.4%为正常，>0.4%为高波。你的判断必须基于此标准。
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
第一反应：
自我质疑：
最终结论：

第七步：矛盾裁决与一致性决策
请以顶尖交易员的身份，独立完成以下裁决流程。你需要处理两类矛盾：一是单个币种内部的多空信号矛盾，二是跨币种之间的宏观方向矛盾。

第一阶段：单币种内部分歧裁决
1. 汇总并标注：逐条列出前六步的所有最终结论，为每条清晰标注看涨或看跌。
2. 独立赋权并说明依据：为每条结论赋予权重（合计100%），并详细解释在当前市场环境下，为何如此分配权重。某些信号被赋予更高权重，是因为它们更贴合当前市场阶段（如趋势市看动量，震荡市看支撑阻力）；某些被压低，是因为其时效性或可靠性存疑。
3. 强制回答三个裁决问题：
   · 最不可忽视的信号是什么？ 在当前市况下，哪一个信号你绝不敢忽视？它为何具备如此决定性的分量？
   · 最强的反向信号是什么？ 与你的初步方向相反的最强证据是哪一个？你究竟是选择压过它，还是被它说服？明确写出你作出这种取舍的核心依据。
   · 核心假设与证伪条件是什么？ 你对这个币种的初步方向判断，是建立在哪一两个核心假设之上的？请清晰列出假设，并定义“证伪条件”——即一旦盘中出现什么具体现象或数据，就说明假设已被推翻，你必须立即改变对该币种的观点。
【单币种裁决后的强制交叉检验】
在完成你的初步方向判断后，必须执行以下检验：
1. 回顾第六步中另一币种的核心指标，判断其整体结构是偏多还是偏空。
2. 若你的初步方向与另一币种的结构方向一致，则通过检验，可继续输出该方向。
3. 若你的初步方向与另一币种的结构方向相反，你必须：
   a) 明确指出矛盾点；
   b) 基于第七步的宏观环境裁判（CVD斜率一致性、ETH/BTC汇率分位、恐慌贪婪动量）来决定最终方向；
   c) 若宏观环境无法明确裁决，则必须输出 neutral。
完成第一阶段的单币种分析后，你将对BTC和ETH分别得出一个初步方向（看涨/看跌）。现在，进入关键的跨币种一致性检验。
---
第二阶段：跨币种强制一致性裁决
触发条件： 若你的分析池中同时包含BTC和ETH的完整六步结论，且第一阶段得出的两者最终方向相反（一个看涨，一个看跌），则必须执行此阶段。否则，跳过此阶段，直接输出第一阶段裁决。
1. 识别矛盾根源：明确指出BTC和ETH在哪个关键信号的解读上出现了根本性分歧（例如：BTC的资金流信号看涨，但ETH的链上活跃度信号看跌；或两者对同一宏观事件的反应权重不同）。
2. 宏观环境终极裁判：你必须在更高维度上解决这对矛盾。请根据以下三个客观维度，判断当前市场的整体风险偏好，你只能选择Risk-On（风险偏好）或Risk-Off（风险规避）：
   · 维度一：CVD（累积成交量增量）斜率一致性：BTC和ETH的CVD斜率方向是否一致？若同向向上，强烈指向Risk-On；同向向下，强烈指向Risk-Off；若背离，此维度得分中性。
   · 维度二：ETH/BTC汇率分位：当前汇率是否处于极端分位（如<10%或>90%）？汇率位于极高或极低位本身是市场情绪极端的信号，需结合背景解读（如极高往往对应疯狂Risk-On末期，极低对应恐慌Risk-Off末期）。
   · 维度三：恐慌贪婪指数动量：该指数是处于上升趋势还是下降趋势？上升趋势支撑Risk-On，下降趋势支撑Risk-Off。
3. 作出裁决并统一方向：基于上述宏观判断，你不再是“二选一”，而是让宏观环境成为最高法官：
   · 若宏观环境判定为 Risk-On，则你必须放弃看跌方向，将所有相关币种的最终方向统一为 看涨。你的逻辑是：在风险偏好回升的市场中，单个币种的利空信号服从于整体市场的资金回暖。
   · 若宏观环境判定为 Risk-Off，则你必须放弃看涨方向，将所有相关币种的最终方向统一为 看跌。你的逻辑是：覆巢之下无完卵，单个币种的利好难以对抗系统性抛压。
   · 若三个维度的信号严重冲突，导致你无法明确判断宏观环境，则必须输出 neutral（观望）。这是一个完全合法的顶级决策，它意味着你承认当前市场存在你无法解决的认知矛盾，选择退场观察。

【短线决策纪律】
你的交易方向必须与你推演的“最先发生的30分钟-4小时内的价格路径”保持一致。
- 如果你预判价格将“先下跌（或回调）再上涨”，且下跌幅度超过当前1小时ATR的0.8倍，则当前不应做多。你可选择：① 等待下跌结束再做多（输出“观望”并设定挂单条件）；② 立即短线做空捕捉那段下跌（输出“做空”，止盈目标为下跌目标位）。
- 同样，若预判“先上涨后下跌”，则不应做空。
- 这条纪律的核心是：你的入场方向必须与推演中首先发生的那段显著运动同向，否则就观望。

【最终裁决前强制决策】
根据你的价格推演，你必须严格执行以下决策树，不得跳过：
1.  确定第一段运动方向： 价格将先 上涨 / 下跌（选择其一）
2.  计算第一段运动的预估幅度：_____ 点
3.  当前1小时ATR的0.8倍（阈值）：_____ 点
4.  决策：
    - 若第一段运动幅度 < 阈值，则方向不受此纪律强制，可自由选择。
    - 若第一段运动幅度 ≥ 阈值，且方向为下跌，则你只能选择： 短线做空 / 观望 + 挂单
    - 若第一段运动幅度 ≥ 阈值，且方向为上涨，则你只能选择： 短线做多 / 观望 + 挂单
    - **绝对不能跨过第一段反向操作！** 若你想操作的方向与第一段运动相反，你必须输出“观望”！
5.  你的选择： （填写：短线做多 / 短线做空 / 观望）

只有在完成上述选择后，你才能输出最终的JSON（不要代码块），并附上完整的推演过程。

---
最终输出标准格式
最终裁决：
· 最终方向：看涨 / 看跌 / 观望
· 核心假设：[列出一两个核心假设]
· 证伪条件：[明确列出盘中会推翻此决策的具体现象或数据]

随后，请完成流动性猎杀推演专业研判：
价格路径推演：必须专业研判，基于当前清算池分布、对手盘结构和资金流方向，描述价格最可能如何测试并触发关键流动性区域，以及触发后可能产生的连锁反应。需包含触发条件和证伪标准。

推理自检：
1. 我的最终裁决是否完全基于前六步的数据和结论？
2. 我在哪一步的“自我质疑”中发现了后来被证实为关键的风险点？
3. 如果我错了，最可能是在哪一步的假设上栽了跟头？
4. 我是否严格遵守了【短线决策纪律】和【最终裁决前强制决策】？若方向与第一段运动相反，我是否已改为观望？

入场区间（说明依据）：
止损位（说明依据）：
止盈位（说明依据）：
主动证伪信号：
微观盘口确认：
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
                max_tokens=8192,
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

    # ==================== 新增强制一致性校验 ====================
    reasoning = s.get("reasoning", "")
    atr_1h = data.get("atr_1h", data.get("atr_15m", 0) * 2) if data else 0

    if direction in ("long", "short") and atr_1h > 0:
        first_leg_down = re.search(r'(?:先.*?下[跌挫].*?再.*?上[涨升])|(?:第一段.*?下跌)', reasoning)
        first_leg_up = re.search(r'(?:先.*?上[涨升].*?再.*?下[跌挫])|(?:第一段.*?上涨)', reasoning)

        if first_leg_down and direction == "long":
            return False, "一致性质检失败: 推演路径先跌后涨，但输出做多，违反短线铁律"
        if first_leg_up and direction == "short":
            return False, "一致性质检失败: 推演路径先涨后跌，但输出做空，违反短线铁律"

        if "不应做多" in reasoning and direction == "long":
            return False, "一致性质检失败: 推理明确声明不应做多，但输出long"
        if "不应做空" in reasoning and direction == "short":
            return False, "一致性质检失败: 推理明确声明不应做空，但输出short"

    final_decision_match = re.search(r'最终方向[：:]\s*(看涨|看跌|观望)', reasoning)
    if final_decision_match:
        decision_text = final_decision_match.group(1)
        inferred = {"看涨": "long", "做多": "long", "看跌": "short", "做空": "short", "观望": "neutral"}.get(decision_text)
        if inferred and inferred != direction:
            return False, f"一致性质检失败: 推理最终方向为{decision_text}({inferred})，但JSON输出为{direction}"

    return True, ""
