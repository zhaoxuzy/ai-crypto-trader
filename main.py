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
    data = client.get_all_data(symbol)  # 获取所有链上/合约数据

    # 补充 OKX 实时价格（如果 CoinGlass 未提供或做交叉验证）
    inst_id = f"{symbol}-USDT-SWAP"
    real_price = get_current_price(inst_id)
    if real_price > 0:
        data["mark_price"] = real_price  # 用 OKX 实时价覆盖
        logger.info(f"OKX 实时价格: {real_price}")

    # 构建 ETH 辅助数据（用于跨币种验证，仅当分析 BTC 时使用）
    eth_data = None
    if symbol == "BTC":
        eth_client = CoinGlassClient()
        eth_data = eth_client.get_all_data("ETH")
        # 如果 eth_data 中的 mark_price 也可用 OKX 修正
        eth_real = get_current_price("ETH-USDT-SWAP")
        if eth_real > 0:
            eth_data["mark_price"] = eth_real

    prompt = build_prompt(data, symbol, eth_data=eth_data)

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
