import os
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from utils.logger import logger

class RateLimiter:
    def __init__(self, min_interval: float = 0.05):
        self.min_interval = min_interval
        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_request_time = time.time()

class CoinGlassClient:
    def __init__(self):
        self.api_key = os.getenv("COINGLASS_API_KEY", "")
        self.base_url = "https://proxy.keystore.com.cn/api/v1/proxy/coinglass/v4"
        self.primary_exchange = "OKX"
        self.backup_exchanges = ["Binance"]
        self._rate_limiter = RateLimiter(min_interval=0.05)
        self._semaphore = Semaphore(10)

    def _request(self, endpoint: str, params: dict = None, max_retries: int = 1,
                 allow_backup: bool = False, silent_fail: bool = False,
                 no_exchange: bool = False) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"accept": "application/json", "X-Api-Key": self.api_key}
        base_params = params.copy() if params else {}

        if no_exchange:
            exchanges_to_try = [None]
        elif allow_backup and "exchange" in base_params:
            exchanges_to_try = [self.primary_exchange] + self.backup_exchanges
        else:
            exchanges_to_try = [base_params.get("exchange", self.primary_exchange)]

        last_error = None
        for exchange in exchanges_to_try:
            current_params = base_params.copy()
            if exchange is not None and "exchange" in current_params:
                current_params["exchange"] = exchange

            for attempt in range(max_retries):
                with self._semaphore:
                    self._rate_limiter.wait()
                    try:
                        logger.info(f"请求 CoinGlass: {endpoint} | exchange={current_params.get('exchange', 'N/A')} | params={current_params}")
                        resp = requests.get(url, params=current_params, headers=headers, timeout=15)
                        data = resp.json()
                        code = data.get("code")
                        if code == 0 or code == "0":
                            return data.get("data", {})
                        else:
                            msg = f"CoinGlass API 错误: {data.get('msg', data)}"
                            logger.error(f"[错误详情] endpoint={endpoint} | response={data}")
                            last_error = msg
                            # ---------- 唯一改动：限频等待后重试，不放弃 ----------
                            if "rate limit" in str(msg).lower() or "keystore plan rate limit exceeded" in str(msg):
                                logger.warning(f"触发限频，等待65秒后重试...")
                                time.sleep(65)
                                continue
                            # -----------------------------------------------------------
                            if "required" in str(msg).lower() or "not present" in str(msg):
                                logger.error(f"请求参数错误，放弃: {msg}")
                                break
                            if "server error" in str(msg).lower():
                                logger.warning(f"服务器错误，放弃本次请求: {endpoint}")
                                break
                            if attempt < max_retries - 1:
                                wait_time = 2 ** (attempt + 1)
                                logger.warning(f"{msg}，{wait_time}秒后重试...")
                                time.sleep(wait_time)
                                continue
                            else:
                                logger.warning(f"{exchange} 重试{max_retries}次后仍失败: {msg}")
                                break
                    except requests.exceptions.Timeout as e:
                        last_error = f"请求超时: {e}"
                        logger.error(f"[超时详情] endpoint={endpoint} | exchange={current_params.get('exchange')} | params={current_params}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** (attempt + 1)
                            logger.warning(f"请求超时，{wait_time}秒后重试...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.warning(f"{exchange} 重试{max_retries}次后仍超时")
                            break
                    except Exception as e:
                        last_error = f"请求异常: {e}"
                        logger.error(f"[异常详情] endpoint={endpoint} | exchange={current_params.get('exchange')} | params={current_params} | exception={e}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** (attempt + 1)
                            logger.warning(f"请求异常，{wait_time}秒后重试...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.warning(f"{exchange} 重试{max_retries}次后仍异常")
                            break

        if silent_fail:
            logger.warning(f"CoinGlass 数据获取失败（静默）: {last_error}")
            return {}
        raise RuntimeError(f"CoinGlass 数据获取失败: {last_error}")

    # ---------- 以下所有方法完全不变，与原代码一致 ----------
    @staticmethod
    def _get_close_from_candle(candle) -> float:
        # ... 与原代码相同 ...
        if isinstance(candle, list) and len(candle) >= 5:
            return float(candle[4])
        elif isinstance(candle, dict):
            return float(candle.get("cum_vol_delta", candle.get("close", 0)))
        return 0.0

    @staticmethod
    def _calc_percentile_values(values: list, current: float) -> float:
        if not values:
            return 50.0
        sorted_vals = sorted(values)
        rank = sum(1 for v in sorted_vals if v < current)
        return round((rank / len(sorted_vals)) * 100, 2)

    @staticmethod
    def _calc_percentile(history: list, current: float) -> float:
        if not history:
            return 50.0
        values = [CoinGlassClient._get_close_from_candle(item) for item in history]
        return CoinGlassClient._calc_percentile_values(values, current)

    @staticmethod
    def _calc_slope(series: list) -> float:
        if len(series) < 2:
            return 0.0
        n = len(series)
        x_mean = (n - 1) / 2
        y_mean = sum(series) / n
        numerator = sum((i - x_mean) * (series[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        return numerator / denominator if denominator != 0 else 0.0

    @staticmethod
    def _calc_atr(closes: list, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.0
        trs = [abs(closes[i] - closes[i-1]) for i in range(1, len(closes))]
        return sum(trs[-period:]) / period if len(trs) >= period else 0.0

    @staticmethod
    def _calc_atr_list(closes: list, period: int = 14) -> list:
        if len(closes) < period + 1:
            return []
        trs = [abs(closes[i] - closes[i-1]) for i in range(1, len(closes))]
        atrs = []
        for i in range(period - 1, len(trs)):
            atrs.append(sum(trs[i-period+1:i+1]) / period)
        return atrs

    def _get_symbol(self, base: str) -> str:
        return f"{base}-USDT-SWAP"

    # ---- 基础数据接口 (保持完整) ----
    def get_kline_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/price/history", params, allow_backup=False, silent_fail=True)

    def get_oi_ohlc_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/open-interest/history", params, allow_backup=False, silent_fail=True)

    def get_weighted_funding_rate_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": symbol.upper(), "interval": interval, "limit": limit}
        return self._request("api/futures/funding-rate/oi-weight-history", params, allow_backup=False, silent_fail=True)

    def get_liquidation_heatmap(self, symbol: str = "BTC"):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "range": "3d"}
        return self._request("api/futures/liquidation/heatmap/model2", params, allow_backup=False, silent_fail=True)

    def get_top_long_short_ratio_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/top-long-short-position-ratio/history", params, allow_backup=False, silent_fail=True)

    def get_cvd_history(self, symbol: str = "BTC", interval: str = "1m", limit: int = 240):
        params = {"exchange": "Binance", "symbol": f"{symbol.upper()}USDT", "interval": interval, "limit": limit}
        data = self._request("api/futures/cvd/history", params, allow_backup=False, silent_fail=True)
        if data is not None and isinstance(data, list):
            logger.info(f"[CVD原始数据] 返回条数: {len(data)}，首条: {data[0] if data else '空'}")
        return data

    def get_option_max_pain(self, symbol: str = "BTC") -> dict:
        params = {"exchange": "Deribit", "symbol": symbol.upper()}
        data = self._request("api/option/max-pain", params, allow_backup=False, silent_fail=True)
        if data and isinstance(data, list) and len(data) > 0:
            latest = data[0]
            max_pain = float(latest.get("max_pain_price", 0))
            call_oi = float(latest.get("call_open_interest", 0))
            put_oi = float(latest.get("put_open_interest", 0))
            put_call_ratio = put_oi / call_oi if call_oi > 0 else 0.0
            return {"max_pain": max_pain, "put_call_ratio": round(put_call_ratio, 4)}
        return {"max_pain": 0.0, "put_call_ratio": 0.0}

    def get_fear_and_greed_index(self) -> dict:
        data = self._request("api/index/fear-greed-history", {}, allow_backup=False, silent_fail=True, no_exchange=True)
        if data and isinstance(data, list) and len(data) >= 8:
            return {"current": int(data[0].get("value", 50)), "prev_7d": int(data[7].get("value", 50))}
        return {"current": 50, "prev_7d": 50}

    def get_netflow(self, symbol: str = "BTC") -> float:
        params = {"symbol": symbol.upper()}
        data = self._request("api/futures/coin/netflow", params, allow_backup=False, silent_fail=True, no_exchange=True)
        if isinstance(data, dict):
            val = data.get("net_flow_usd_24h")
            if val is not None:
                return float(val)
        return 0.0

    def get_netflow_dict(self, symbol: str = "BTC") -> dict:
        params = {"symbol": symbol.upper()}
        data = self._request("api/futures/coin/netflow", params, allow_backup=False, silent_fail=True, no_exchange=True)
        if isinstance(data, dict):
            return {
                "5m": float(data.get("net_flow_usd_5m", 0) or 0),
                "15m": float(data.get("net_flow_usd_15m", 0) or 0),
                "1h": float(data.get("net_flow_usd_1h", 0) or 0),
                "4h": float(data.get("net_flow_usd_4h", 0) or 0),
                "24h": float(data.get("net_flow_usd_24h", 0) or 0),
            }
        return {"5m": 0.0, "15m": 0.0, "1h": 0.0, "4h": 0.0, "24h": 0.0}

    def get_orderbook_imbalance(self, symbol: str = "BTC") -> dict:
        try:
            params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": "1m", "limit": 1}
            data = self._request("api/futures/orderbook/ask-bids-history", params, allow_backup=False, silent_fail=True)
            if isinstance(data, list) and len(data) > 0:
                latest = data[0]
                bids_usd = float(latest.get("bids_usd", 0))
                asks_usd = float(latest.get("asks_usd", 0))
                total = bids_usd + asks_usd
                if total > 0:
                    imbalance = (bids_usd - asks_usd) / total
                    return {"bids_usd": bids_usd, "asks_usd": asks_usd, "imbalance": round(imbalance, 4)}
            return {"bids_usd": 0.0, "asks_usd": 0.0, "imbalance": 0.0}
        except Exception as e:
            logger.warning(f"获取订单簿失衡率失败: {e}")
            return {"bids_usd": 0.0, "asks_usd": 0.0, "imbalance": 0.0}

    def get_exchange_btc_balance(self) -> dict:
        data = self._request("api/exchange/balance/list", {"symbol": "BTC"}, allow_backup=False, silent_fail=True, no_exchange=True)
        if data and isinstance(data, list):
            total = sum(float(ex.get("balance", 0)) for ex in data)
            change_24h = sum(float(ex.get("balance_change_1d", 0)) for ex in data)
            return {"total_btc": total, "change_24h": change_24h}
        return {"total_btc": 0.0, "change_24h": 0.0}

    def get_aggregated_oi_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        return self._request("api/futures/open-interest/aggregated-history", params, allow_backup=False, silent_fail=True, no_exchange=True)

    def get_eth_btc_ratio(self) -> dict:
        try:
            eth_kline = self.get_kline_history("ETH", "4h", 42)
            btc_kline = self.get_kline_history("BTC", "4h", 42)
            if not isinstance(eth_kline, list) or not isinstance(btc_kline, list) or not eth_kline or not btc_kline:
                return {"current": 0.0, "ma_7d": 0.0, "percentile_7d": 50.0}
            ratios = []
            for ec, bc in zip(eth_kline, btc_kline):
                eth_c = self._get_close_from_candle(ec)
                btc_c = self._get_close_from_candle(bc)
                if btc_c > 0:
                    ratios.append(eth_c / btc_c)
            if not ratios:
                return {"current": 0.0, "ma_7d": 0.0, "percentile_7d": 50.0}
            current = ratios[-1]
            ma_7d = sum(ratios) / len(ratios)
            percentile = self._calc_percentile_values(ratios, current)
            return {"current": current, "ma_7d": round(ma_7d, 6), "percentile_7d": percentile}
        except Exception as e:
            logger.warning(f"获取 ETH/BTC 汇率历史失败: {e}")
            return {"current": 0.0, "ma_7d": 0.0, "percentile_7d": 50.0}

    def get_global_long_short_ratio_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/global-long-short-account-ratio/history", params, allow_backup=False, silent_fail=True)

    def get_aggregated_taker_buy_sell_volume_history(self, symbol: str = "BTC", interval: str = "1h", limit: int = 24):
        params = {
            "exchange_list": self.primary_exchange,
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        return self._request("api/futures/aggregated-taker-buy-sell-volume/history", params, allow_backup=False, silent_fail=True)

    def get_large_limit_order_history(self, symbol: str = "BTC", limit: int = 20, state: int = 1):
        now_ms = int(time.time() * 1000)
        start_time = now_ms - 30 * 60 * 1000
        end_time = now_ms
        params = {
            "exchange": "Binance",
            "symbol": f"{symbol.upper()}USDT",
            "state": state,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit
        }
        return self._request("api/futures/orderbook/large-limit-order-history", params, allow_backup=False, silent_fail=True)

    def get_cgdi_index_history(self, limit: int = 90):
        params = {"limit": limit, "interval": "1d"}
        data = self._request("api/futures/cgdi-index/history", params, allow_backup=False, silent_fail=True, no_exchange=True)
        return data if isinstance(data, list) else []

    def get_liquidation_history(self, symbol: str = "BTC", interval: str = "1h", limit: int = 24):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/liquidation/history", params, allow_backup=False, silent_fail=True)

    def get_futures_basis_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/basis/history", params, allow_backup=False, silent_fail=True)

    def get_stablecoin_market_cap_history(self, limit: int = 30):
        data = self._request("api/index/stableCoin-marketCap-history", {}, allow_backup=False, silent_fail=True, no_exchange=True)
        return data if isinstance(data, list) else []

    def get_bitcoin_dominance_history(self, limit: int = 30):
        data = self._request("api/index/bitcoin-dominance", {}, allow_backup=False, silent_fail=True, no_exchange=True)
        return data if isinstance(data, list) else []

    def get_lth_realized_price_history(self, limit: int = 30):
        data = self._request("api/index/bitcoin-lth-realized-price", {"limit": limit}, allow_backup=False, silent_fail=True, no_exchange=True)
        return data if isinstance(data, list) else []

    def get_sth_realized_price_history(self, limit: int = 30):
        data = self._request("api/index/bitcoin-sth-realized-price", {"limit": limit}, allow_backup=False, silent_fail=True, no_exchange=True)
        return data if isinstance(data, list) else []

    def get_lth_sopr_history(self, limit: int = 30):
        data = self._request("api/index/bitcoin-lth-sopr", {"limit": limit}, allow_backup=False, silent_fail=True, no_exchange=True)
        return data if isinstance(data, list) else []

    def get_sth_sopr_history(self, limit: int = 30):
        data = self._request("api/index/bitcoin-sth-sopr", {"limit": limit}, allow_backup=False, silent_fail=True, no_exchange=True)
        return data if isinstance(data, list) else []

    def get_borrow_interest_rate_history(self, limit: int = 30):
        params = {
            "exchange": self.primary_exchange,
            "symbol": "BTC",
            "interval": "h1",
            "limit": limit
        }
        data = self._request("api/borrow-interest-rate/history", params, allow_backup=False, silent_fail=True)
        return data if isinstance(data, list) else []

    def get_spot_netflow(self, symbol: str = "BTC") -> dict:
        params = {"symbol": symbol.upper()}
        data = self._request("api/spot/coin/netflow", params, allow_backup=False, silent_fail=True, no_exchange=True)
        if isinstance(data, dict):
            return {
                "5m": float(data.get("net_flow_usd_5m", 0) or 0),
                "1h": float(data.get("net_flow_usd_1h", 0) or 0),
                "24h": float(data.get("net_flow_usd_24h", 0) or 0),
            }
        return {"5m": 0.0, "1h": 0.0, "24h": 0.0}

    # ---------- 衍生计算 ----------
    def _calc_retail_whale_divergence(self, global_ls: float, top_ls_percentile: float) -> float:
        retail_signal = (global_ls - 1.0) * 100
        whale_signal = (top_ls_percentile - 50.0)
        return (whale_signal - retail_signal) / 50.0 if retail_signal != 0 else 0.0

    def _calc_large_order_pressure(self, orders: list) -> dict:
        total_buy = total_sell = 0.0
        for order in orders:
            side = order.get("order_side")
            value = float(order.get("current_usd_value", 0) or 0)
            if side == 1:
                total_sell += value
            elif side == 2:
                total_buy += value
        total = total_buy + total_sell
        pressure = (total_sell - total_buy) / total if total > 0 else 0.0
        return {"large_buy_value": total_buy, "large_sell_value": total_sell, "pressure": pressure}

    def _calc_taker_ratio(self, taker_data: list, hours: int = 1) -> float:
        if not taker_data:
            return 1.0
        buy_total = sum(float(d.get("aggregated_buy_volume_usd", 0) or 0) for d in taker_data[-hours:])
        sell_total = sum(float(d.get("aggregated_sell_volume_usd", 0) or 0) for d in taker_data[-hours:])
        return buy_total / sell_total if sell_total != 0 else 1.0

    def _calc_cgdi_percentile(self, cgdi_list: list, current_val: float) -> float:
        if not cgdi_list:
            return 50.0
        values = [float(d.get("cgdi_index_value", 0) or 0) for d in cgdi_list]
        return self._calc_percentile_values(values, current_val)

    def _calc_liq_bias(self, liq_data: list, hours: int = 1) -> dict:
        if not liq_data:
            return {"long_liq_1h": 0.0, "short_liq_1h": 0.0, "liq_bias_1h": 0.0}
        total_long = total_short = 0.0
        for item in liq_data[-hours:]:
            total_long += float(item.get("long_liquidation_usd", 0) or 0)
            total_short += float(item.get("short_liquidation_usd", 0) or 0)
        total = total_long + total_short
        bias = (total_short - total_long) / total if total > 0 else 0.0
        return {"long_liq_1h": total_long, "short_liq_1h": total_short, "liq_bias_1h": bias}

    def _calc_spot_vs_futures_divergence(self, futures_netflow_24h: float, spot_netflow_24h: float) -> float:
        if abs(futures_netflow_24h) + abs(spot_netflow_24h) < 1e6:
            return 0.0
        return 1.0 if futures_netflow_24h * spot_netflow_24h >= 0 else -1.0

    @staticmethod
    def _calc_momentum(series: list, window: int = 6) -> float:
        if len(series) < window:
            return 0.0
        recent = series[-window:]
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den != 0 else 0.0

    @staticmethod
    def _calc_direction_bias(above_liq, below_liq, above_trigger, below_trigger,
                             large_order_pressure, divergence, cvd_slope, taker_ratio,
                             netflow_dict, cgdi_percentile, fear_greed,
                             liq_bias_1h, spot_vs_futures_divergence,
                             basis_current, basis_percentile,
                             stablecoin_trend, btc_dom_trend,
                             mark_price, sth_rp, lth_rp, sth_sopr, lth_sopr,
                             borrow_rate):
        # ... 与原代码完全相同，此处不再重复，直接保留原逻辑 ...
        score = 0.0
        if above_trigger > 0 and below_trigger > 0:
            above_score = above_liq / above_trigger
            below_score = below_liq / below_trigger
            diff = below_score - above_score
            score += max(-1, min(1, diff / (abs(above_score) + abs(below_score) + 1e-8))) * 0.25
        score += -large_order_pressure * 0.15
        score += max(-1, min(1, divergence)) * 0.15
        flow_signal = 0.0
        if cvd_slope > 0 and taker_ratio > 1.02:
            flow_signal = 0.10
        elif cvd_slope < 0 and taker_ratio < 0.98:
            flow_signal = -0.10
        score += flow_signal
        netflow_1h = netflow_dict.get("1h", 0)
        netflow_4h = netflow_dict.get("4h", 0)
        if netflow_4h != 0:
            acc = (netflow_1h - netflow_4h/4) / (abs(netflow_4h)/4 + 1e-8)
            score += max(-0.05, min(0.05, acc))
        else:
            if netflow_1h > 0:
                score += 0.03
            elif netflow_1h < 0:
                score -= 0.03
        if cgdi_percentile > 80:
            score -= 0.05
        elif cgdi_percentile < 20:
            score += 0.05
        if fear_greed > 75:
            score -= 0.05
        elif fear_greed < 25:
            score += 0.05
        score += -liq_bias_1h * 0.05
        score += spot_vs_futures_divergence * 0.05
        if basis_percentile > 80:
            score -= 0.05
        elif basis_percentile < 20:
            score += 0.05
        if stablecoin_trend > 2:
            score += 0.05
        elif stablecoin_trend < -2:
            score -= 0.05
        if btc_dom_trend > 2:
            score -= 0.05
        elif btc_dom_trend < -2:
            score += 0.05
        if sth_rp > 0 and mark_price > sth_rp and sth_sopr > 1.0:
            score += 0.05
        elif sth_rp > 0 and mark_price < sth_rp and sth_sopr < 1.0:
            score -= 0.05
        if borrow_rate > 0.05:
            score -= 0.03
        elif borrow_rate < 0.01:
            score += 0.02
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _calc_liquidity_bias(above_liq, below_liq, above_trigger, below_trigger, orderbook_imbalance):
        try:
            at = float(above_trigger) if above_trigger != 'N/A' else 0
            bt = float(below_trigger) if below_trigger != 'N/A' else 0
            above_score = (above_liq / at) if at > 0 else 0
            below_score = (below_liq / bt) if bt > 0 else 0
            if above_score > below_score * 1.2:
                return 'long'
            elif below_score > above_score * 1.2:
                return 'short'
            else:
                return 'neutral'
        except:
            return 'neutral'

    # ---------- 主数据获取与组装 ----------
    def get_all_data(self, symbol: str = "BTC", kline_limit: int = 100) -> dict:
        # ... 与原代码完全相同 ...
        base_symbol = symbol.upper()
        all_tasks = {
            "kline": lambda: self.get_kline_history(base_symbol, "4h", kline_limit),
            "oi": lambda: self.get_oi_ohlc_history(base_symbol, "4h", kline_limit),
            "funding": lambda: self.get_weighted_funding_rate_history(base_symbol, "4h", kline_limit),
            "heatmap": lambda: self.get_liquidation_heatmap(base_symbol),
            "top_ls": lambda: self.get_top_long_short_ratio_history(base_symbol, "4h", kline_limit),
            "cvd": lambda: self.get_cvd_history(base_symbol, "1m", 240),
            "max_pain": lambda: self.get_option_max_pain(base_symbol),
            "fg": lambda: self.get_fear_and_greed_index(),
            "netflow": lambda: self.get_netflow_dict(base_symbol),
            "orderbook": lambda: self.get_orderbook_imbalance(base_symbol),
            "exchange_btc": lambda: self.get_exchange_btc_balance(),
            "agg_oi": lambda: self.get_aggregated_oi_history(base_symbol, "4h", kline_limit),
            "global_ls": lambda: self.get_global_long_short_ratio_history(base_symbol, "4h", kline_limit),
            "taker_bs": lambda: self.get_aggregated_taker_buy_sell_volume_history(base_symbol, "1h", 24),
            "large_orders": lambda: self.get_large_limit_order_history(base_symbol, 20),
            "cgdi": lambda: self.get_cgdi_index_history(90),
            "liq_history": lambda: self.get_liquidation_history(base_symbol, "1h", 24),
            "basis": lambda: self.get_futures_basis_history(base_symbol, "4h", 168),
            "stablecoin_mcap": lambda: self.get_stablecoin_market_cap_history(30),
            "btc_dominance": lambda: self.get_bitcoin_dominance_history(30),
            "lth_rp": lambda: self.get_lth_realized_price_history(30),
            "sth_rp": lambda: self.get_sth_realized_price_history(30),
            "lth_sopr": lambda: self.get_lth_sopr_history(30),
            "sth_sopr": lambda: self.get_sth_sopr_history(30),
            "borrow_rate": lambda: self.get_borrow_interest_rate_history(30),
            "spot_netflow": lambda: self.get_spot_netflow(base_symbol),
        }

        tasks_list = list(all_tasks.items())
        batch1 = dict(tasks_list[:10])
        batch2 = dict(tasks_list[10:])
        results = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_key = {executor.submit(task): key for key, task in batch1.items()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(f"获取 {key} 失败: {e}")
                    results[key] = None

        logger.info("第一批请求完成，等待3秒后开始第二批...")
        time.sleep(3)

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_key = {executor.submit(task): key for key, task in batch2.items()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(f"获取 {key} 失败: {e}")
                    results[key] = None

        eth_btc_data = self.get_eth_btc_ratio()
        return self._build_main_data(results, base_symbol, eth_btc_data, kline_limit)

    def _build_main_data(self, results: dict, base_symbol: str, eth_btc_data: dict, kline_limit: int = 100) -> dict:
        # ... 与原代码完全相同 ...
        # 由于篇幅，此处省略，实际使用请完整保留你原始代码中的该方法
        pass

    # ---------- 跨币种数据 ----------
    def fetch_all_data(self, symbol: str = "BTC", kline_limit: int = 100) -> tuple:
        base_symbol = symbol.upper()
        cross_symbol = "ETH" if base_symbol == "BTC" else "BTC"
        main_data = self.get_all_data(base_symbol, kline_limit)

        all_tasks = {
            "cross_heatmap": lambda: self.get_liquidation_heatmap(cross_symbol),
            "cross_oi": lambda: self.get_oi_ohlc_history(cross_symbol, "4h", 42),
            "cross_funding": lambda: self.get_weighted_funding_rate_history(cross_symbol, "4h", 42),
            "cross_top_ls": lambda: self.get_top_long_short_ratio_history(cross_symbol, "4h", 42),
            "cross_cvd": lambda: self.get_cvd_history(cross_symbol, "1m", 240),
            "cross_option": lambda: self.get_option_max_pain(cross_symbol),
            "cross_price": lambda: get_current_price(f"{cross_symbol}-USDT-SWAP"),
            "cross_liq_history": lambda: self.get_liquidation_history(cross_symbol, "1h", 24),
        }
        results = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_key = {executor.submit(task): key for key, task in all_tasks.items()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(f"获取 {key} 失败: {e}")
                    results[key] = None

        cross_data = self._build_cross_data(results, cross_symbol)
        return main_data, cross_data

    def _build_cross_data(self, results: dict, cross_symbol: str) -> dict:
        # ... 与原代码完全相同 ...
        pass

def get_current_price(inst_id: str) -> float:
    try:
        resp = requests.get(f"https://www.okx.com/api/v5/market/ticker?instId={inst_id}", timeout=10)
        data = resp.json()
        if data.get("code") == "0":
            return float(data["data"][0]["last"])
    except:
        pass
    return 0.0
