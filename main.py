#!/usr/bin/env python3
""" 主程序：获取数据 -> AI分析 -> 校验 -> 推送钉钉 """
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_client.deepseek import build_prompt, call_deepseek, validate_strategy
from notifier.dingtalk import format_strategy_message, send_dingtalk_message
from utils.logger import logger

# 示例数据（请替换为你的CoinGlass真实数据获取函数）
def get_sample_data(symbol="ETH"):
    """返回示例数据字典。请替换为实际API调用。"""
    return {
        "timestamp": "2026-04-23T14:00:00Z",
        "mark_price": 2345.0,
        "atr_15m": 3.8, "atr_1h": 7.5,
        "vol_factor": 0.96, "price_percentile": 78.0,
        "above_liq": 1200e9, "below_liq": 1550e9,
        "above_cluster": "2400-2450", "below_cluster": "2250-2300",
        "liq_ratio": 1200/1550,
        "orderbook_bids": 85e6, "orderbook_asks": 92e6, "orderbook_imbalance": -0.04,
        "funding_rate": 0.0032, "funding_percentile": 70.0,
        "oi": 2.5e9, "oi_percentile": 65.0, "oi_change_24h": -5.2,
        "agg_oi": 5.0e9, "agg_oi_change_24h": -3.8,
        "top_ls_ratio": 0.88, "top_ls_percentile": 95.0,
        "fear_greed": 50, "fear_greed_prev_7d": 52,
        "max_pain": 2375.0, "put_call_ratio": 1.42,
        "cvd_slope": 105000.0, "netflow": 450e6, "exchange_btc_change_24h": -500,
        "eth_btc_ratio": 0.0301, "eth_btc_ma_7d": 0.0308, "eth_btc_percentile": 2.0,
        "data_quality": {}
    }

def main():
    symbol = os.getenv("STRATEGY_SYMBOL", "ETH").upper()
    logger.info(f"===== 策略生成流程开始 ({symbol}) =====")

    # 1. 获取数据（需替换为你的真实数据获取逻辑）
    #    如果同时分析BTC，可获取eth_data后传入build_prompt
    data = get_sample_data(symbol)
    # 示例：模拟 eth_data（如果 symbol == "BTC" 则可传入）
    eth_data = None
    if symbol == "BTC":
        eth_data = get_sample_data("ETH")

    # 2. 构建 Prompt
    prompt = build_prompt(data, symbol, eth_data=eth_data)

    # 3. 调用 AI
    try:
        strategy = call_deepseek(prompt)
    except Exception as e:
        logger.error(f"{symbol} 策略生成失败: {e}")
        return

    # 4. 校验
    valid, msg = validate_strategy(strategy, data)
    if not valid:
        logger.error(f"策略校验失败: {msg}")
        return

    # 5. 格式化并推送
    markdown_msg = format_strategy_message(symbol, strategy, data)
    if send_dingtalk_message(markdown_msg, title=f"{symbol} 策略推送"):
        logger.info("信号已成功推送到钉钉")
    else:
        logger.error("推送失败")

if __name__ == "__main__":
    main()
