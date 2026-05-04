import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_client.deepseek import build_prompt, call_trader, validate_strategy
from notifier.dingtalk import format_strategy_message, send_dingtalk_message
from data.fetcher import CoinGlassClient, get_current_price
from utils.logger import logger

def main():
    symbol = os.getenv("STRATEGY_SYMBOL", "ETH").upper()
    logger.info(f"===== 策略生成流程开始 ({symbol}) =====")

    client = CoinGlassClient()
    
    # 获取数据
    try:
        data, cross_data = client.fetch_all_data(symbol)
    except Exception as e:
        logger.error(f"数据获取失败: {e}")
        return

    # 补充实时价格
    inst_id = f"{symbol}-USDT-SWAP"
    real_price = get_current_price(inst_id)
    if real_price > 0:
        data["mark_price"] = real_price
        logger.info(f"OKX 实时价格: {real_price}")

    logger.info(f"跨币种数据完整性: {cross_data.get('_complete')}")

    # 构建 Prompt
    prompt = build_prompt(data, symbol, eth_data=cross_data)

    # 交易员深度分析并输出策略
    try:
        strategy = call_trader(prompt)
    except Exception as e:
        logger.error(f"策略生成失败: {e}")
        return

    # 硬编码校验
    valid, msg = validate_strategy(strategy, data)
    if not valid:
        logger.error(f"策略校验失败: {msg}")
        return

    # 直接推送最终策略
    final_msg = format_strategy_message(symbol, strategy, data)
    send_dingtalk_message(final_msg, title=f"{symbol} 策略推送")

    logger.info("策略已推送至钉钉")

if __name__ == "__main__":
    main()