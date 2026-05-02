import os
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from utils.logger import logger


class RateLimiter:
    """线程安全的轻量限速器（0.1秒间隔）"""
    def __init__(self, min_interval: float = 0.1):
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
        self._rate_limiter = RateLimiter(min_interval=0.1)
        self._semaphore = Semaphore(8)

    def _request(self, endpoint: str, params: dict = None, max_retries: int = 3,
                 allow_backup: bool = True, silent_fail: bool = False,
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
                            last_error = msg
                            if "rate limit" in str(msg).lower() or "keystore plan rate limit exceeded" in str(msg):
                                logger.warning(f"触发限频，放弃本次请求: {endpoint}")
                                break
                            if "required" in str(msg).lower() or "not present" in str(msg):
                                logger.error(f"请求参数错误，放弃: {msg}")
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

    # ---------- 基础工具函数 ----------
    @staticmethod
    def _get_close_from_candle(candle) -> float:
        if isinstance(candle, list) and len(candle) >= 5:
            return float(candle[4])
        elif isinstance(candle, dict):
            return float(candle.get("cum_vol_delta", candle.get("close", 0)))
        return 0.0

    @staticmethod
    def _calc_percentile(history: list, current: float) -> float:
        if not history:
            return 50.0
        values = [CoinGlassClient._get_close_from_candle(item) for item in history]
        values.sort()
        rank = sum(1 for v in values if v < current)
        return round((rank / len(values)) * 100, 2)

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
        """生成 OKX 永续合约格式的 symbol，如 ETH-USDT-SWAP"""
        return f"{base}-USDT-SWAP"

    # ========== 原有接口 ==========
    def get_kline_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/price/history", params, allow_backup=True, silent_fail=True)

    def get_oi_ohlc_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/open-interest/history", params, allow_backup=True, silent_fail=True)

    def get_weighted_funding_rate_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        # 此接口文档中 symbol 格式为基础币种，如 BTC，无需 USDT 后缀
        params = {"exchange": self.primary_exchange, "symbol": symbol.upper(), "interval": interval, "limit": limit}
        return self._request("api/futures/funding-rate/oi-weight-history", params, allow_backup=False, silent_fail=True)

    def get_liquidation_heatmap(self, symbol: str = "BTC"):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "range": "3d"}
        return self._request("api/futures/liquidation/heatmap/model2", params, allow_backup=True, silent_fail=True)

    def get_top_long_short_ratio_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/top-long-short-position-ratio/history", params, allow_backup=True, silent_fail=True)

    def get_cvd_history(self, symbol: str = "BTC", interval: str = "1m", limit: int = 240):
        # Binance 专用的 CVD 接口，symbol 格式为 BTCUSDT
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
            data = self._request("api/futures/orderbook/ask-bids-history", params, allow_backup=True, silent_fail=True)
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
            if not eth_kline or not btc_kline:
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
            sorted_r = sorted(ratios)
            rank = sum(1 for r in sorted_r if r < current)
            percentile = round((rank / len(sorted_r)) * 100, 2)
            return {"current": current, "ma_7d": round(ma_7d, 6), "percentile_7d": percentile}
        except Exception as e:
            logger.warning(f"获取 ETH/BTC 汇率历史失败: {e}")
            return {"current": 0.0, "ma_7d": 0.0, "percentile_7d": 50.0}

    # ---------- 其他接口 ----------
    def get_global_long_short_ratio_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        # 官方文档示例：exchange=Binance&symbol=BTCUSDT，但此处使用主交易所 OKX，故用 OKX 格式
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/global-long-short-account-ratio/history", params, allow_backup=True, silent_fail=True)

    def get_aggregated_taker_buy_sell_volume_history(self, symbol: str = "BTC", interval: str = "1h", limit: int = 24):
        params = {
        "exchange": self.primary_exchange,
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }
    return self._request("api/futures/aggregated-taker-buy-sell-volume/history", params, allow_backup=True, silent_fail=True)

    def get_large_limit_order_history(self, symbol: str = "BTC", limit: int = 20):
        params = {"symbol": f"{symbol.upper()}USDT", "limit": limit}
        return self._request("api/futures/orderbook/large-limit-order-history", params, allow_backup=False, silent_fail=True, no_exchange=True)

    def get_cgdi_index_history(self, limit: int = 90):
        params = {"limit": limit, "interval": "1d"}
        data = self._request("api/futures/cgdi-index/history", params, allow_backup=False, silent_fail=True, no_exchange=True)
        return data if isinstance(data, list) else []

    def get_liquidation_history(self, symbol: str = "BTC", interval: str = "1h", limit: int = 24):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/liquidation/history", params, allow_backup=True, silent_fail=True)

    def get_futures_basis_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        # 官方文档示例：exchange=Binance&symbol=BTCUSDT，但此处使用主交易所 OKX
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/basis/history", params, allow_backup=True, silent_fail=True)

    def get_stablecoin_market_cap_history(self, limit: int = 30):
        params = {"exchange_list": "USDT", "limit": limit}
        data = self._request("api/index/stableCoin-marketCap-history", params, allow_backup=False, silent_fail=True)
        return data if isinstance(data, list) else []

    def get_bitcoin_dominance_history(self, limit: int = 30):
        data = self._request("api/index/bitcoin-dominance", {"limit": limit}, allow_backup=False, silent_fail=True, no_exchange=True)
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
        data = self._request("api/borrow-interest-rate/history", {"limit": limit}, allow_backup=False, silent_fail=True, no_exchange=True)
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

    # ========== 衍生计算 ==========
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
        values.sort()
        rank = sum(1 for v in values if v < current_val)
        return round((rank / len(values)) * 100, 2)

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

    # ========== 主数据获取与组装 ==========
    def get_all_data(self, symbol: str = "BTC", kline_limit: int = 100) -> dict:
        base_symbol = symbol.upper()
        tasks = {
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
        results = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_key = {executor.submit(task): key for key, task in tasks.items()}
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
        kline_data = results.get("kline", [])
        oi_data = results.get("oi", [])
        funding_data = results.get("funding", [])
        top_ls_data = results.get("top_ls", [])
        cvd_data = results.get("cvd", [])
        heatmap_raw = results.get("heatmap", {})
        max_pain_data = results.get("max_pain", {})
        fg_data = results.get("fg", {"current": 50, "prev_7d": 50})
        netflow_dict = results.get("netflow", {})
        orderbook = results.get("orderbook", {})
        exchange_btc = results.get("exchange_btc", {})
        agg_oi_data = results.get("agg_oi", [])
        global_ls_data = results.get("global_ls", [])
        taker_bs_data = results.get("taker_bs", [])
        large_orders_data = results.get("large_orders", [])
        cgdi_data = results.get("cgdi", [])
        liq_history_data = results.get("liq_history", [])
        basis_data = results.get("basis", [])
        stablecoin_mcap_data = results.get("stablecoin_mcap", [])
        btc_dom_data = results.get("btc_dominance", [])
        lth_rp_data = results.get("lth_rp", [])
        sth_rp_data = results.get("sth_rp", [])
        lth_sopr_data = results.get("lth_sopr", [])
        sth_sopr_data = results.get("sth_sopr", [])
        borrow_rate_data = results.get("borrow_rate", [])
        spot_netflow_data = results.get("spot_netflow", {})

        data_quality = {key: "✅" if results.get(key) else "❌ 缺失" for key in results.keys()}

        mark_price = self._get_close_from_candle(kline_data[-1]) if kline_data else 0.0
        closes = [self._get_close_from_candle(k) for k in kline_data]
        atr_4h = self._calc_atr(closes, 14) if len(closes) >= 14 else 0.0
        atr_list = self._calc_atr_list(closes, 14)
        avg_atr_7d = sum(atr_list) / len(atr_list) if atr_list else 0.0
        vol_factor = atr_4h / avg_atr_7d if avg_atr_7d > 0 else 1.0
        price_percentile = self._calc_percentile(kline_data, mark_price)
        atr_15m = atr_4h * 0.25 if atr_4h > 0 else 0.0
        atr_1h_val = atr_4h * 0.5
        atr_1h_ratio = (atr_1h_val / mark_price) * 100 if mark_price > 0 else 0.0

        above_liq, below_liq, above_cluster, below_cluster, liq_ratio = 0, 0, "N/A", "N/A", 0.0
        if heatmap_raw:
            y_axis = heatmap_raw.get("y_axis", [])
            liq_data = heatmap_raw.get("liquidation_leverage_data", [])
            pain_map = {}
            for item in liq_data:
                if isinstance(item, list) and len(item) >= 3:
                    price = float(y_axis[int(item[1])]) if int(item[1]) < len(y_axis) else 0
                    intensity = float(item[2])
                    if price > mark_price: above_liq += intensity
                    elif price < mark_price: below_liq += intensity
                    pain_map[price] = intensity
            liq_ratio = above_liq / below_liq if below_liq > 0 else 0.0
            if pain_map:
                above_prices = [p for p in pain_map if p > mark_price]
                below_prices = [p for p in pain_map if p < mark_price]
                if above_prices:
                    max_above = max(above_prices, key=lambda p: pain_map[p])
                    above_cluster = f"{max_above*0.99:.0f}-{max_above*1.01:.0f}"
                if below_prices:
                    max_below = max(below_prices, key=lambda p: pain_map[p])
                    below_cluster = f"{max_below*0.99:.0f}-{max_below*1.01:.0f}"

        above_trigger_str = f"{float(above_cluster.split('-')[0]) - mark_price:+.0f}" if above_cluster != 'N/A' and '-' in above_cluster else "N/A"
        below_trigger_str = f"{mark_price - float(below_cluster.split('-')[1]):+.0f}" if below_cluster != 'N/A' and '-' in below_cluster else "N/A"
        above_trigger_val = float(above_trigger_str) if above_trigger_str != 'N/A' else 0
        below_trigger_val = float(below_trigger_str) if below_trigger_str != 'N/A' else 0

        oi_current = self._get_close_from_candle(oi_data[-1]) if oi_data else 0.0
        oi_percentile = self._calc_percentile(oi_data, oi_current)
        oi_change_24h = 0.0
        if len(oi_data) >= 6:
            oi_24h_ago = self._get_close_from_candle(oi_data[-6])
            oi_change_24h = (oi_current - oi_24h_ago) / oi_24h_ago * 100 if oi_24h_ago > 0 else 0.0

        funding_current = self._get_close_from_candle(funding_data[-1]) if funding_data else 0.0
        funding_percentile = self._calc_percentile(funding_data, funding_current)

        top_ls_current = 0.0
        if top_ls_data and isinstance(top_ls_data, list) and len(top_ls_data) > 0:
            latest = top_ls_data[-1]
            if isinstance(latest, dict) and "top_position_long_short_ratio" in latest:
                top_ls_current = float(latest.get("top_position_long_short_ratio", 0))
            else:
                top_ls_current = self._get_close_from_candle(latest)
        top_ls_percentile = self._calc_percentile(top_ls_data, top_ls_current) if top_ls_data else 50.0

        cvd_series = [self._get_close_from_candle(c) for c in cvd_data] if cvd_data else []
        cvd_slope = self._calc_slope(cvd_series)

        agg_oi_current = self._get_close_from_candle(agg_oi_data[-1]) if agg_oi_data else 0.0
        agg_oi_change_24h = 0.0
        if len(agg_oi_data) >= 6:
            agg_oi_24h_ago = self._get_close_from_candle(agg_oi_data[-6])
            agg_oi_change_24h = (agg_oi_current - agg_oi_24h_ago) / agg_oi_24h_ago * 100 if agg_oi_24h_ago > 0 else 0.0

        fear_greed = fg_data.get("current", 50)
        fear_greed_prev_7d = fg_data.get("prev_7d", 50)

        cvd_acceleration = self._calc_momentum(cvd_series[-60:]) if len(cvd_series) >= 60 else 0.0
        oi_series = [self._get_close_from_candle(c) for c in oi_data] if oi_data else []
        oi_acceleration = self._calc_momentum(oi_series[-30:]) if len(oi_series) >= 30 else 0.0
        funding_series = [self._get_close_from_candle(c) for c in funding_data] if funding_data else []
        funding_momentum = self._calc_momentum(funding_series[-30:]) if len(funding_series) >= 30 else 0.0

        global_ls_current = 0.0
        if global_ls_data and isinstance(global_ls_data, list) and len(global_ls_data) > 0:
            latest_gls = global_ls_data[-1]
            if isinstance(latest_gls, dict):
                global_ls_current = float(latest_gls.get("long_short_ratio", 1.0))
            elif isinstance(latest_gls, list):
                global_ls_current = self._get_close_from_candle(latest_gls)
            else:
                global_ls_current = 1.0
        retail_whale_divergence = self._calc_retail_whale_divergence(global_ls_current, top_ls_percentile)

        taker_ratio_1h = self._calc_taker_ratio(taker_bs_data, hours=1)

        large_order_info = self._calc_large_order_pressure(large_orders_data if large_orders_data else [])
        large_order_pressure = large_order_info.get("pressure", 0.0)
        large_buy_value = large_order_info.get("large_buy_value", 0.0)
        large_sell_value = large_order_info.get("large_sell_value", 0.0)

        cgdi_current = 0.0
        if cgdi_data and len(cgdi_data) > 0:
            cgdi_current = float(cgdi_data[-1].get("cgdi_index_value", 1000) or 1000)
        cgdi_percentile = self._calc_cgdi_percentile(cgdi_data, cgdi_current)

        liq_bias_info = self._calc_liq_bias(liq_history_data if liq_history_data else [], hours=1)
        long_liq_1h = liq_bias_info.get("long_liq_1h", 0.0)
        short_liq_1h = liq_bias_info.get("short_liq_1h", 0.0)
        liq_bias_1h = liq_bias_info.get("liq_bias_1h", 0.0)

        # 新增指标计算
        basis_current = 0.0
        basis_percentile = 50.0
        if basis_data and len(basis_data) > 0:
            basis_values = [self._get_close_from_candle(b) for b in basis_data]
            basis_current = basis_values[-1]
            basis_percentile = self._calc_percentile(basis_data, basis_current)

        stablecoin_trend = 0.0
        stablecoin_mcap_current = 0.0
        if stablecoin_mcap_data and len(stablecoin_mcap_data) > 0:
            mcap_values = [float(d.get("value", 0) or 0) for d in stablecoin_mcap_data]
            stablecoin_mcap_current = mcap_values[-1]
            if len(mcap_values) >= 7:
                stablecoin_trend = (mcap_values[-1] - mcap_values[-7]) / (mcap_values[-7] + 1) * 100

        btc_dom_current = 0.0
        btc_dom_trend = 0.0
        if btc_dom_data and len(btc_dom_data) > 0:
            dom_values = [float(d.get("value", 0) or 0) for d in btc_dom_data]
            btc_dom_current = dom_values[-1]
            if len(dom_values) >= 7:
                btc_dom_trend = (dom_values[-1] - dom_values[-7]) / (dom_values[-7] + 1) * 100

        lth_rp = 0.0
        if lth_rp_data and len(lth_rp_data) > 0:
            lth_rp = float(lth_rp_data[-1].get("value", 0) or 0)

        sth_rp = 0.0
        if sth_rp_data and len(sth_rp_data) > 0:
            sth_rp = float(sth_rp_data[-1].get("value", 0) or 0)

        lth_sopr = 1.0
        if lth_sopr_data and len(lth_sopr_data) > 0:
            lth_sopr = float(lth_sopr_data[-1].get("value", 1.0) or 1.0)

        sth_sopr = 1.0
        if sth_sopr_data and len(sth_sopr_data) > 0:
            sth_sopr = float(sth_sopr_data[-1].get("value", 1.0) or 1.0)

        borrow_rate_current = 0.0
        if borrow_rate_data and len(borrow_rate_data) > 0:
            borrow_rate_current = float(borrow_rate_data[-1].get("value", 0) or 0)

        spot_netflow_24h = spot_netflow_data.get("24h", 0.0) if isinstance(spot_netflow_data, dict) else 0.0
        spot_netflow_1h = spot_netflow_data.get("1h", 0.0) if isinstance(spot_netflow_data, dict) else 0.0

        spot_vs_futures_divergence = self._calc_spot_vs_futures_divergence(
            netflow_dict.get("24h", 0.0), spot_netflow_24h
        )

        direction_bias = self._calc_direction_bias(
            above_liq, below_liq, above_trigger_val, below_trigger_val,
            large_order_pressure,
            retail_whale_divergence,
            cvd_slope, taker_ratio_1h,
            netflow_dict,
            cgdi_percentile,
            fear_greed,
            liq_bias_1h,
            spot_vs_futures_divergence,
            basis_current, basis_percentile,
            stablecoin_trend,
            btc_dom_trend,
            mark_price, sth_rp, lth_rp, sth_sopr, lth_sopr,
            borrow_rate_current,
        )

        liquidity_bias = self._calc_liquidity_bias(above_liq, below_liq, above_trigger_val, below_trigger_val, orderbook.get("imbalance", 0.0))

        lure_risk_factor = 0.0
        try:
            if orderbook.get("imbalance", 0) < -0.1 and below_trigger_val < above_trigger_val:
                lure_risk_factor = 0.6
            elif orderbook.get("imbalance", 0) > 0.1 and above_trigger_val < below_trigger_val:
                lure_risk_factor = 0.6
        except:
            pass

        eth_btc_ratio = eth_btc_data.get("current", 0.0)
        eth_btc_ma_7d = eth_btc_data.get("ma_7d", 0.0)
        eth_btc_percentile = eth_btc_data.get("percentile_7d", 50.0)

        return {
            "mark_price": mark_price,
            "atr": atr_4h,
            "atr_15m": atr_15m,
            "atr_1h": atr_1h_val,
            "atr_1h_ratio": round(atr_1h_ratio, 2),
            "vol_factor": vol_factor,
            "price_percentile": price_percentile,
            "above_liq": above_liq,
            "below_liq": below_liq,
            "liq_ratio": liq_ratio,
            "above_cluster": above_cluster,
            "below_cluster": below_cluster,
            "above_trigger": above_trigger_str,
            "below_trigger": below_trigger_str,
            "max_pain": max_pain_data.get("max_pain", 0.0),
            "put_call_ratio": max_pain_data.get("put_call_ratio", 0.0),
            "top_ls_ratio": top_ls_current,
            "top_ls_percentile": top_ls_percentile,
            "funding_rate": funding_current,
            "funding_percentile": funding_percentile,
            "oi": oi_current,
            "oi_percentile": oi_percentile,
            "oi_change_24h": oi_change_24h,
            "agg_oi": agg_oi_current,
            "agg_oi_change_24h": agg_oi_change_24h,
            "cvd_mean": sum(cvd_series) / len(cvd_series) / 1e6 if cvd_series else 0.0,
            "cvd_slope": cvd_slope,
            "cvd_acceleration": cvd_acceleration,
            "oi_acceleration": oi_acceleration,
            "funding_momentum": funding_momentum,
            "fear_greed": fear_greed,
            "fear_greed_prev_7d": fear_greed_prev_7d,
            "eth_btc_ratio": eth_btc_ratio,
            "eth_btc_ma_7d": eth_btc_ma_7d,
            "eth_btc_percentile": eth_btc_percentile,
            "netflow": netflow_dict.get("24h", 0.0),
            "netflow_5m": netflow_dict.get("5m", 0.0),
            "netflow_1h": netflow_dict.get("1h", 0.0),
            "netflow_24h": netflow_dict.get("24h", 0.0),
            "orderbook_bids": orderbook.get("bids_usd", 0.0),
            "orderbook_asks": orderbook.get("asks_usd", 0.0),
            "orderbook_imbalance": orderbook.get("imbalance", 0.0),
            "exchange_btc_total": exchange_btc.get("total_btc", 0.0),
            "exchange_btc_change_24h": exchange_btc.get("change_24h", 0.0),
            "data_quality": data_quality,
            "liquidity_bias": liquidity_bias,
            "lure_risk_factor": lure_risk_factor,
            "direction_bias": direction_bias,
            "retail_whale_divergence": retail_whale_divergence,
            "global_ls_ratio": global_ls_current,
            "taker_ratio_1h": taker_ratio_1h,
            "large_order_pressure": large_order_pressure,
            "large_buy_value": large_buy_value,
            "large_sell_value": large_sell_value,
            "cgdi_current": cgdi_current,
            "cgdi_percentile": cgdi_percentile,
            "long_liq_1h": long_liq_1h,
            "short_liq_1h": short_liq_1h,
            "liq_bias_1h": liq_bias_1h,
            "basis_current": basis_current,
            "basis_percentile": basis_percentile,
            "stablecoin_mcap": stablecoin_mcap_current,
            "stablecoin_trend_7d": stablecoin_trend,
            "btc_dominance": btc_dom_current,
            "btc_dominance_trend_7d": btc_dom_trend,
            "lth_realized_price": lth_rp,
            "sth_realized_price": sth_rp,
            "lth_sopr": lth_sopr,
            "sth_sopr": sth_sopr,
            "borrow_rate": borrow_rate_current,
            "spot_netflow_1h": spot_netflow_1h,
            "spot_netflow_24h": spot_netflow_24h,
            "spot_vs_futures_divergence": spot_vs_futures_divergence,
        }

    @staticmethod
    def _calc_direction_bias(above_liq, below_liq, above_trigger, below_trigger,
                             large_order_pressure, divergence, cvd_slope, taker_ratio,
                             netflow_dict, cgdi_percentile, fear_greed,
                             liq_bias_1h, spot_vs_futures_divergence,
                             basis_current, basis_percentile,
                             stablecoin_trend, btc_dom_trend,
                             mark_price, sth_rp, lth_rp, sth_sopr, lth_sopr,
                             borrow_rate):
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

    # ========== 跨币种数据 ==========
    def fetch_all_data(self, symbol: str = "BTC", kline_limit: int = 100) -> tuple:
        base_symbol = symbol.upper()
        cross_symbol = "ETH" if base_symbol == "BTC" else "BTC"
        main_data = self.get_all_data(base_symbol, kline_limit)

        tasks = {
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
            future_to_key = {executor.submit(task): key for key, task in tasks.items()}
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
        data = {}
        complete = True
        price = results.get("cross_price")
        if price is not None and price > 0:
            mark_price = price
            data["mark_price"] = mark_price
        else:
            mark_price = 0.0
            data["mark_price"] = 0.0
            complete = False

        heatmap = results.get("cross_heatmap")
        if heatmap:
            y_axis = heatmap.get("y_axis", [])
            liq_data = heatmap.get("liquidation_leverage_data", [])
            above_liq, below_liq = 0, 0
            for item in liq_data:
                if isinstance(item, list) and len(item) >= 3:
                    price_i = float(y_axis[int(item[1])]) if int(item[1]) < len(y_axis) else 0
                    intensity = float(item[2])
                    if price_i > mark_price: above_liq += intensity
                    elif price_i < mark_price: below_liq += intensity
            data["above_liq"] = above_liq
            data["below_liq"] = below_liq
            data["liq_ratio"] = above_liq / below_liq if below_liq > 0 else 0.0
        else:
            data["above_liq"] = data["below_liq"] = data["liq_ratio"] = 0.0
            complete = False

        oi = results.get("cross_oi")
        if oi:
            oi_current = self._get_close_from_candle(oi[-1])
            data["oi_percentile"] = self._calc_percentile(oi, oi_current)
            oi_change = 0.0
            if len(oi) >= 6:
                prev = self._get_close_from_candle(oi[-6])
                oi_change = (oi_current - prev) / prev * 100 if prev > 0 else 0.0
            data["oi_change_24h"] = oi_change
        else:
            data["oi_percentile"] = 50.0
            data["oi_change_24h"] = 0.0
            complete = False

        funding = results.get("cross_funding")
        if funding:
            current = self._get_close_from_candle(funding[-1])
            data["funding_rate"] = current
            data["funding_percentile"] = self._calc_percentile(funding, current)
        else:
            data["funding_rate"] = 0.0
            data["funding_percentile"] = 50.0
            complete = False

        top_ls = results.get("cross_top_ls")
        if top_ls:
            latest = top_ls[-1]
            if isinstance(latest, dict) and "top_position_long_short_ratio" in latest:
                current = float(latest.get("top_position_long_short_ratio", 0))
            else:
                current = self._get_close_from_candle(latest)
            data["top_ls_ratio"] = current
            data["top_ls_percentile"] = self._calc_percentile(top_ls, current)
        else:
            data["top_ls_ratio"] = 0.0
            data["top_ls_percentile"] = 50.0
            complete = False

        cvd = results.get("cross_cvd")
        if cvd:
            series = [self._get_close_from_candle(c) for c in cvd]
            data["cvd_slope"] = self._calc_slope(series)
        else:
            data["cvd_slope"] = 0.0
            complete = False

        option = results.get("cross_option")
        if option:
            data["put_call_ratio"] = option.get("put_call_ratio", 0.0)
            data["max_pain"] = option.get("max_pain", 0.0)

        liq_history = results.get("cross_liq_history")
        if liq_history:
            liq_info = self._calc_liq_bias(liq_history, hours=1)
            data["liq_bias_1h"] = liq_info.get("liq_bias_1h", 0.0)
        else:
            data["liq_bias_1h"] = 0.0
            complete = False

        data["_complete"] = complete
        return data


def get_current_price(inst_id: str) -> float:
    try:
        resp = requests.get(f"https://www.okx.com/api/v5/market/ticker?instId={inst_id}", timeout=10)
        data = resp.json()
        if data.get("code") == "0":
            return float(data["data"][0]["last"])
    except:
        pass
    return 0.0
