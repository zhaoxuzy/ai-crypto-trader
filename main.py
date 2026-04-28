import os, sys, time, threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_client.deepseek import build_prompt, call_deepseek, validate_strategy, call_reviewer, call_judge, apply_final_verdict
from notifier.dingtalk import format_strategy_message, format_review_message, format_judge_message, send_dingtalk_message, format_final_decision
from data.fetcher import CoinGlassClient, get_current_price
from utils.logger import logger

def main():
    symbol = os.getenv("STRATEGY_SYMBOL", "ETH").upper()
    logger.info(f"===== 策略生成流程开始 ({symbol}) =====")

    client = CoinGlassClient()
    
    # 一次性获取主数据 + 跨币种数据（合并并发，避免串行限速）
    try:
        data, cross_data = client.fetch_all_data(symbol)
    except Exception as e:
        logger.error(f"数据获取失败: {e}")
        return

    # 补充 OKX 实时价格（如果 fetch_all_data 里已获取过则不会重复，但这里做一次备用）
    inst_id = f"{symbol}-USDT-SWAP"
    real_price = get_current_price(inst_id)
    if real_price > 0:
        data["mark_price"] = real_price
        logger.info(f"OKX 实时价格: {real_price}")

    # 构建 Prompt（跨币种数据传入 eth_data）
    prompt = build_prompt(data, symbol, eth_data=cross_data)

    # 1. 首席交易员生成策略
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

    # 3. 异步审查：审查官B → 法官C，在线程内即时推送
    def run_review_and_judge():
        nonlocal strategy

        # --- 审查官B ---
        try:
            reviewer_report = call_reviewer(strategy, data, symbol)
            # 推送审查报告
            review_msg = format_review_message(symbol, strategy, reviewer_report, data)
            send_dingtalk_message(review_msg, title=f"{symbol} 策略推送 (风控审计)")
        except Exception as e:
            logger.warning(f"审查官B调用失败: {e}")
            strategy["_reviewed"] = False
            return

        # --- 法官C ---
        try:
            judge_result = call_judge(strategy, reviewer_report, data, symbol)
            strategy = apply_final_verdict(strategy, judge_result)
            # 推送最终裁决（使用新模板，展示原始块并加粗关键字）
            judge_msg = format_final_decision(symbol, strategy, judge_result, data)
            send_dingtalk_message(judge_msg, title=f"{symbol} 策略推送 (交易委员会裁决)")
        except Exception as e:
            logger.warning(f"法官C调用失败: {e}")
            strategy["_reviewed"] = False
            return

        # 存储审查报告和法官裁决理由供后续使用（可选）
        strategy["_reviewer_report"] = reviewer_report.get("full_report", "")
        strategy["_judge_reasoning"] = judge_result.get("judge_C", {}).get("reasoning", "")

    review_thread = threading.Thread(target=run_review_and_judge)
    review_thread.start()
    review_thread.join(timeout=600)  # 总超时 10分钟，足够 B 和 C 完成

    # 4. 超时保护：如果线程仍未结束，降级执行原始策略
    if review_thread.is_alive():
        logger.warning("审查超时，按原策略降级执行")
        strategy["_reviewed"] = False
        strategy["_review_timeout"] = True
        size_map = {"heavy": "medium", "medium": "light", "light": "light"}
        strategy["position_size"] = size_map.get(strategy.get("position_size", "light"), "light")
        # 推送降级后的最终信号
        final_msg = format_strategy_message(symbol, strategy, data)
        final_msg = "> ⚠️ **审查超时，已按原策略降级执行**\n\n" + final_msg
        send_dingtalk_message(final_msg, title=f"{symbol} 策略推送 (最终)")

    logger.info("所有信号已推送至钉钉")

if __name__ == "__main__":
    main()
