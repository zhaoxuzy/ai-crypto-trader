import os, sys, time, threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_client.deepseek import build_prompt, call_deepseek, validate_strategy, call_reviewer, call_judge, apply_final_verdict
from notifier.dingtalk import format_strategy_message, send_dingtalk_message
from data.fetcher import CoinGlassClient, get_current_price
from utils.logger import logger

def main():
    symbol = os.getenv("STRATEGY_SYMBOL", "ETH").upper()
    logger.info(f"===== 策略生成流程开始 ({symbol}) =====")

    client = CoinGlassClient()
    data = client.get_all_data(symbol)

    inst_id = f"{symbol}-USDT-SWAP"
    real_price = get_current_price(inst_id)
    if real_price > 0:
        data["mark_price"] = real_price
        logger.info(f"OKX 实时价格: {real_price}")

    cross_symbol = "ETH" if symbol == "BTC" else "BTC"
    cross_data = None
    try:
        cross_data = client.get_cross_asset_data(cross_symbol)
    except Exception as e:
        logger.warning(f"获取跨币种数据失败：{e}")

    prompt = build_prompt(data, symbol, eth_data=cross_data)

    # 1. 交易员A生成策略
    try:
        strategy = call_deepseek(prompt)
    except Exception as e:
        logger.error(f"{symbol} 策略生成失败: {e}")
        return

    # 基础校验
    valid, msg = validate_strategy(strategy, data)
    if not valid:
        logger.error(f"策略校验失败: {msg}")
        return

    # 2. 推送初步信号
    preliminary_strategy = strategy.copy()
    preliminary_strategy["_preliminary"] = True
    prelim_msg = format_strategy_message(symbol, preliminary_strategy, data)
    send_dingtalk_message(prelim_msg, title=f"{symbol} 策略推送 (审查中...)")

    # 3. 审查官B + 法官C 异步执行
    def run_review_and_judge():
        nonlocal strategy
        try:
            reviewer_report = call_reviewer(strategy, data, symbol)
        except Exception as e:
            logger.warning(f"审查官B调用失败: {e}")
            strategy["_reviewed"] = False
            return

        try:
            judge_result = call_judge(strategy, reviewer_report, data, symbol)
        except Exception as e:
            logger.warning(f"法官C调用失败: {e}")
            strategy["_reviewed"] = False
            return

        strategy = apply_final_verdict(strategy, judge_result)

    review_thread = threading.Thread(target=run_review_and_judge)
    review_thread.start()
    review_thread.join(timeout=90)

    # 4. 超时保护
    if review_thread.is_alive():
        logger.warning("审查超时，按原策略降级执行")
        strategy["_reviewed"] = False
        strategy["_review_timeout"] = True
        size_map = {"heavy": "medium", "medium": "light", "light": "light"}
        strategy["position_size"] = size_map.get(strategy.get("position_size", "light"), "light")

    # 5. 最终校验与推送
    valid, msg = validate_strategy(strategy, data)
    if not valid:
        logger.error(f"最终策略校验失败: {msg}")
        return

    final_msg = format_strategy_message(symbol, strategy, data)
    if strategy.get("_review_timeout"):
        final_msg = "> ⚠️ **审查超时，按原策略降级执行**\n\n" + final_msg
    send_dingtalk_message(final_msg, title=f"{symbol} 策略推送 (最终)")
    logger.info("最终信号已推送至钉钉")

if __name__ == "__main__":
    main()
