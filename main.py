import sys, os, threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_client.deepseek import (build_prompt, call_trader, validate_strategy,
                                call_reviewer, call_judge, apply_final_verdict, _force_neutral)
from notifier.dingtalk import (format_strategy_message, format_review_message,
                               format_final_decision, send_dingtalk_message)
from data.fetcher import CoinGlassClient, get_current_price
from utils.logger import logger

def main():
    symbol = os.getenv("STRATEGY_SYMBOL", "ETH").upper()
    logger.info(f"===== 策略生成流程开始 ({symbol}) =====")

    client = CoinGlassClient()
    try:
        data, cross_data = client.fetch_all_data(symbol)
    except Exception as e:
        logger.error(f"数据获取失败: {e}")
        return

    # 锚点可信度标记
    data_quality = data.get("data_quality", {})
    missing_count = sum(1 for v in data_quality.values() if v == "❌ 缺失")
    if missing_count == 0:
        data["_bias_quality"] = "reliable"
    elif missing_count <= 2:
        data["_bias_quality"] = "degraded"
    else:
        data["_bias_quality"] = "untrusted"

    inst_id = f"{symbol}-USDT-SWAP"
    real_price = get_current_price(inst_id)
    if real_price > 0:
        data["mark_price"] = real_price
        logger.info(f"OKX 实时价格: {real_price}")

    logger.info(f"跨币种数据完整性: {cross_data.get('_complete')}")

    prompt = build_prompt(data, symbol, eth_data=cross_data)

    try:
        strategy = call_trader(prompt)
    except Exception as e:
        logger.error(f"策略生成失败: {e}")
        return

    valid, msg = validate_strategy(strategy, data)
    if not valid:
        logger.error(f"策略校验失败: {msg}")
        return

    # 发送初步方案
    prelim_msg = format_strategy_message(symbol, strategy, data)
    send_dingtalk_message(prelim_msg, title=f"策略｜{symbol} 初步方案 ⏳")

    def run_review_and_judge():
        nonlocal strategy
        try:
            reviewer_report = call_reviewer(strategy, data, symbol)
            review_msg = format_review_message(symbol, strategy, reviewer_report, data)
            send_dingtalk_message(review_msg, title=f"策略｜{symbol} 审计报告 📋")
        except Exception as e:
            logger.warning(f"审计官调用异常: {e}")
            strategy["_reviewed"] = False
            return

        try:
            judge_result = call_judge(strategy, reviewer_report, data, symbol)
            strategy = apply_final_verdict(strategy, judge_result)
            final_msg = format_final_decision(symbol, strategy, judge_result, data)
            send_dingtalk_message(final_msg, title=f"策略｜{symbol} 最终计划 ⚖️")
        except Exception as e:
            logger.warning(f"委员会调用异常: {e}")
            _force_neutral(strategy, "委员会调用失败，强制观望")
            fake_judge = {"final_verdict":"推翻","final_direction":"neutral","final_confidence":"low","final_position_size":"none","entry_price_low":0,"entry_price_high":0,"stop_loss":0,"take_profit":0,"final_reasoning":"委员会失败","weighted_signal":{},"audit_responses":[]}
            final_msg = format_final_decision(symbol, strategy, fake_judge, data)
            send_dingtalk_message(final_msg, title=f"策略｜{symbol} 最终计划 ⚖️")

    review_thread = threading.Thread(target=run_review_and_judge)
    review_thread.start()
    review_thread.join(timeout=600)

    if review_thread.is_alive():
        logger.warning("审查超时，按原策略降级执行")
        strategy["_reviewed"] = False
        size_map = {"heavy": "medium", "medium": "light", "light": "light"}
        strategy["position_size"] = size_map.get(strategy.get("position_size", "light"), "light")
        final_msg = format_strategy_message(symbol, strategy, data)
        final_msg = "> ⚠️ **审查超时，已按原策略降级执行**\n\n" + final_msg
        send_dingtalk_message(final_msg, title=f"策略｜{symbol} 最终计划 ⚠️")

    logger.info("所有信号已推送至钉钉")

if __name__ == "__main__":
    main()
