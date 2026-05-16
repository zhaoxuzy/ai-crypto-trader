import os
import time
import requests
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from utils.logger import logger


class RateLimiter:
    """全局限流器：最小间隔 + 每分钟最大请求数"""
    def __init__(self, min_interval: float = 0.05, max_per_minute: int = 26):
        self.min_interval = min_interval
        self.max_per_minute = max_per_minute
        self._last_request_time = 0.0
        self._window_start = time.time()
        self._window_count = 0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            if now - self._window_start >= 60.0:
                self._window_start = now
                self._window_count = 0
            if self._window_count >= self.max_per_minute:
                sleep_time = 60.0 - (now - self._window_start) + 0.5
                logger.warning(f"本地分钟配额({self.max_per_minute})已用尽，等待 {sleep_time:.1f} 秒...")
                time.sleep(sleep_time)
                now = time.time()
                self._window_start = now
                self._window_count = 0
            elapsed = now - self._last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_request_time = time.time()
            self._window_count += 1


class CoinGlassClient:
    def __init__(self):
        self.api_key = os.getenv("COINGLASS_API_KEY", "")
        self.base_url = "https://proxy.keystore.com.cn/api/v1/proxy/coinglass/v4"
        self.primary_exchange = "OKX"
        self.backup_exchanges = ["Binance"]
        self._rate_limiter = RateLimiter(min_interval=0.05, max_per_minute=26)
        self._semaphore = Semaphore(10)

    # ---------- 内部：OKX 公开 K 线获取 ----------
    def _get_okx_kline(self, symbol: str, interval: str = "4h", limit: int = 168) -> list:
        try:
            bar_map = {"4h": "4H", "1h": "1H", "1m": "1m", "5m": "5m", "15m": "15m"}
            bar = bar_map.get(interval, interval)
            inst_id = f"{symbol.upper()}-USDT-SWAP"
            url = "https://www.okx.com/api/v5/market/candles"
            resp = requests.get(url, params={"instId": inst_id, "bar": bar, "limit": limit}, timeout=10)
            data = resp.json()
            if data.get("code") == "0":
                candles = data["data"]
                candles.reverse()
                result = []
                for c in candles:
                    ts, o, h, l, cl, vol = int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                    result.append([ts, o, h, l, cl, vol])
                return result
        except Exception as e:
            logger.warning(f"OKX 公开 K 线获取失败 ({symbol} {interval}): {e}")
        return []

    # ---------- 内部：恐惧贪婪指数（替代 keystore） ----------
    def get_fear_and_greed_index(self) -> dict:
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=8", timeout=10)
            data = resp.json().get("data", [])
            if len(data) >= 8:
                return {
                    "current": int(data[0].get("value", 50)),
                    "prev_7d": int(data[7].get("value", 50))
                }
        except Exception as e:
            logger.warning(f"获取恐惧贪婪指数失败: {e}")
        return {"current": 50, "prev_7d": 50}

    # ---------- 核心请求方法（限频重试） ----------
    def _request(self, endpoint: str, params: dict = None, max_retries: int = 3,
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

                            if "rate limit" in str(msg).lower() or "keystore plan rate limit exceeded" in str(msg):
                                wait_seconds = 65
                                logger.warning(f"触发限频，等待 {wait_seconds} 秒后重试...")
                                time.sleep(wait_seconds)
                                continue

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

    # ---------- 通用工具 ----------
    @staticmethod
    def _get_close_from_candle(candle) -> float:
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

    # ---------- K 线（优先用 OKX 公开 API） ----------
    def get_kline_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        if self.primary_exchange == "OKX":
            kline = self._get_okx_kline(symbol, interval, limit)
            if kline:
                return kline
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/price/history", params, allow_backup=False, silent_fail=True)

    # ---------- 其余所有接口（与之前保持一致） ----------
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

    # ---------- 衍生计算（保持不变） ----------
    # ...（此处省略后续计算方法和数据组装方法，它们与原始代码完全一致，请直接复用你之前的完整版本）
