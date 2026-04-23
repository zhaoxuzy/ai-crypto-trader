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
    data = client.get_all_data(symbol)

    # 补充 OKX 实时价格
    inst_id = f"{symbol}-USDT-SWAP"
    real_price = get_current_price(inst_id)
    if real_price > 0:
        data["mark_price"] = real_price
        logger.info(f"OKX 实时价格: {real_price}")

    # 获取跨币种验证数据（精简版）
    cross_symbol = "ETH" if symbol == "BTC" else "BTC"
    cross_data = None
    try:
        cross_data = client.get_cross_asset_data(cross_symbol)
    except Exception as e:
        logger.warning(f"获取跨币种数据失败：{e}，将跳过第六步验证")

    # 构建prompt（跨币种数据传入 eth_data 参数，因为字段兼容）
   prompt = build_prompt(data, symbol, eth_data=cross_data)

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
