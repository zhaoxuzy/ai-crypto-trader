import os, sys, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_client.deepseek import build_prompt, call_deepseek, validate_strategy
from notifier.dingtalk import format_strategy_message, send_dingtalk_message
from data.fetcher import CoinGlassClient, get_current_price
from utils.logger import logger

def main():
    symbol = os.getenv("STRATEGY_SYMBOL", "ETH").upper()
    logger.info(f"===== 策略生成流程开始 ({symbol}) =====")

    client = CoinGlassClient()
    data = client.get_all_data(symbol, kline_limit=100)

    inst_id = f"{symbol}-USDT-SWAP"
    real_price = get_current_price(inst_id)
    if real_price > 0:
        data["mark_price"] = real_price
        logger.info(f"OKX 实时价格: {real_price}")

    # 跨币种对比基准
    if symbol == "BTC":
        cross_symbol = "ETH"
    elif symbol == "ETH":
        cross_symbol = "BTC"
    elif symbol == "SOL":
        cross_symbol = "BTC"
    else:
        cross_symbol = None

    cross_data = None
    if cross_symbol:
        try:
            cross_data = client.get_cross_asset_data(cross_symbol)
        except Exception as e:
            logger.warning(f"获取跨币种数据失败：{e}，将跳过第六步验证")

    prompt = build_prompt(data, symbol, eth_data=cross_data, cross_symbol=cross_symbol)

    try:
        strategy = call_deepseek(prompt)
    except Exception as e:
        logger.error(f"{symbol} 策略生成失败: {e}")
        return

    valid, msg = validate_strategy(strategy, data)
    if not valid:
        logger.error(f"策略校验失败: {msg}")
        return

    markdown_msg = format_strategy_message(symbol, strategy, data)
    if send_dingtalk_message(markdown_msg, title=f"{symbol} 策略推送"):
        logger.info("信号已推送至钉钉")
    else:
        logger.error("推送失败")

if __name__ == "__main__":
    main()
