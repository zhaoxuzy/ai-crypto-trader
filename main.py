import os, sys, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_client.deepseek import build_prompt, call_deepseek, validate_strategy, call_devils_advocate
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

    # 构建 prompt（跨币种数据传入 eth_data 参数）
    prompt = build_prompt(data, symbol, eth_data=cross_data)

    # 1. 首次AI分析（前六步+第七步裁决）
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

    # 2. 异议审查官独立质检
    logger.info("启动异议审查官质检...")
    strategy = call_devils_advocate(strategy, data, symbol, cross_data)

    # 对审查后的策略再次进行基础校验
    valid, msg = validate_strategy(strategy, data)
    if not valid:
        logger.error(f"审查后策略校验失败: {msg}")
        return

    # 记录审查结果
    if strategy.get("_reviewed"):
        verdict = strategy.get("_review_verdict", "维持原判")
        logger.info(f"审查完成，判决：{verdict}")
    else:
        logger.warning("审查官未能完成审查，使用原策略")

    # 格式化推送到钉钉
    markdown_msg = format_strategy_message(symbol, strategy, data)
    if send_dingtalk_message(markdown_msg, title=f"{symbol} 策略推送"):
        logger.info("信号已推送至钉钉")
    else:
        logger.error("推送失败")

if __name__ == "__main__":
    main()
