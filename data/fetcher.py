import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from utils.logger import logger


class RateLimiter:
    def __init__(self, min_interval: float = 1.5):
        self.min_interval = min_interval
        self._last_request_time = 0.0

    def wait(self):
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
        self._rate_limiter = RateLimiter(min_interval=1.5)
        self._semaphore = Semaphore(8)

    def _request(self, endpoint: str, params: dict = None, max_retries: int = 3, allow_backup: bool = True, silent_fail: bool = False) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"accept": "application/json", "X-Api-Key": self.api_key}
        base_params = params.copy() if params else {}
        
        if allow_backup and "exchange" in base_params:
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
                        if data.get("code") in (0, "0"):
                            return data.get("data", {})
                        else:
                            msg = f"CoinGlass API 错误: {data.get('msg', data)}"
                            last_error = msg
                            if attempt < max_retries - 1:
                                if "rate limit" in str(msg).lower():
                                    wait_time = 10 * (attempt + 1)
                                else:
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
        return f"{base}-USDT-SWAP"

    def get_kline_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/price/history", params, allow_backup=True, silent_fail=True)

    def get_oi_ohlc_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/open-interest/history", params, allow_backup=True, silent_fail=True)

    def get_weighted_funding_rate_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": symbol.upper(), "interval": interval, "limit": limit}
        return self._request("api/futures/funding-rate/oi-weight-history", params, allow_backup=False, silent_fail=True)

    def get_liquidation_heatmap(self, symbol: str = "BTC"):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "range": "3d"}
        return self._request("api/futures/liquidation/heatmap/model2", params, allow_backup=True, silent_fail=True)

    def get_top_long_short_ratio_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        return self._request("api/futures/top-long-short-position-ratio/history", params, allow_backup=True, silent_fail=True)

    def get_cvd_history(self, symbol: str = "BTC", interval: str = "1m", limit: int = 240):
        params = {"exchange": self.primary_exchange, "symbol": self._get_symbol(symbol), "interval": interval, "limit": limit}
        data = self._request("api/futures/cvd/history", params, allow_backup=True, silent_fail=True)
        if data is not None:
            if isinstance(data, list):
                logger.info(f"[CVD原始数据] 返回条数: {len(data)}，首条: {data[0] if data else '空'}")
            else:
                logger.info(f"[CVD原始数据] 返回类型: {type(data)}，内容: {data}")
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
        data = self._request("api/index/fear-greed-history", {}, allow_backup=False, silent_fail=True)
        if data and isinstance(data, list) and len(data) >= 8:
            return {
                "current": int(data[0].get("value", 50)),
                "prev_7d": int(data[7].get("value", 50))
            }
        return {"current": 50, "prev_7d": 50}

    def get_netflow(self, symbol: str = "BTC") -> float:
        params = {"symbol": symbol.upper()}
        data = self._request("api/futures/coin/netflow", params, allow_backup=False, silent_fail=True)
        logger.info(f"[Netflow原始数据] 返回内容: {data}")
        if isinstance(data, dict):
            for field in ["net_flow_usd_24h", "netflow_24h", "netflow", "netFlow", "flow"]:
                if field in data:
                    val = data.get(field)
                    if val is not None:
                        logger.info(f"✅ 期货资金净流获取成功: {val} (字段: {field})")
                        return float(val)
            logger.warning(f"⚠️ 期货资金净流返回数据中无已知字段，原始数据: {data}")
            return 0.0
        elif isinstance(data, list) and len(data) > 0:
            latest = data[0]
            if isinstance(latest, dict):
                for field in ["net_flow_usd_24h", "netflow_24h", "netflow", "flow", "value"]:
                    if field in latest:
                        return float(latest.get(field, 0))
            logger.warning(f"⚠️ 期货资金净流返回数组，无法解析，原始数据: {data[:2]}")
        logger.warning(f"⚠️ 期货资金净流返回类型异常: {type(data)}")
        return 0.0

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
        data = self._request("api/exchange/balance/list", {"symbol": "BTC"}, allow_backup=False, silent_fail=True)
        if data and isinstance(data, list):
            total = sum(float(ex.get("balance", 0)) for ex in data)
            change_24h = sum(float(ex.get("balance_change_1d", 0)) for ex in data)
            return {"total_btc": total, "change_24h": change_24h}
        return {"total_btc": 0.0, "change_24h": 0.0}

    def get_aggregated_oi_history(self, symbol: str = "BTC", interval: str = "4h", limit: int = 168):
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        return self._request("api/futures/open-interest/aggregated-history", params, allow_backup=False, silent_fail=True)

    def get_eth_btc_ratio(self) -> dict:
        """返回 ETH/BTC 汇率的当前值、7日均值、7日分位数"""
        try:
            eth_kline = self.get_kline_history("ETH", "4h", 42)
            btc_kline = self.get_kline_history("BTC", "4h", 42)

            if not eth_kline or not btc_kline:
                return {"current": 0.0, "ma_7d": 0.0, "percentile_7d": 50.0}

            ratios = []
            for eth_candle, btc_candle in zip(eth_kline, btc_kline):
                eth_close = self._get_close_from_candle(eth_candle)
                btc_close = self._get_close_from_candle(btc_candle)
                if btc_close > 0:
                    ratios.append(eth_close / btc_close)

            if not ratios:
                return {"current": 0.0, "ma_7d": 0.0, "percentile_7d": 50.0}

            current = ratios[-1]
            ma_7d = sum(ratios) / len(ratios)

            sorted_ratios = sorted(ratios)
            rank = sum(1 for r in sorted_ratios if r < current)
            percentile = round((rank / len(sorted_ratios)) * 100, 2)

            return {
                "current": current,
                "ma_7d": round(ma_7d, 6),
                "percentile_7d": percentile
            }
        except Exception as e:
            logger.warning(f"获取 ETH/BTC 汇率历史失败: {e}")
            return {"current": 0.0, "ma_7d": 0.0, "percentile_7d": 50.0}

    def get_all_data(self, symbol: str = "BTC") -> dict:
        base_symbol = symbol.upper()

        tasks = {
            "kline": lambda: self.get_kline_history(base_symbol, "4h", 168),
            "oi": lambda: self.get_oi_ohlc_history(base_symbol, "4h", 168),
            "funding": lambda: self.get_weighted_funding_rate_history(base_symbol, "4h", 168),
            "heatmap": lambda: self.get_liquidation_heatmap(base_symbol),
            "top_ls": lambda: self.get_top_long_short_ratio_history(base_symbol, "4h", 168),
            "cvd": lambda: self.get_cvd_history(base_symbol, "1m", 240),
            "max_pain": lambda: self.get_option_max_pain(base_symbol),
            "fg": lambda: self.get_fear_and_greed_index(),
            "netflow": lambda: self.get_netflow(base_symbol),
            "orderbook": lambda: self.get_orderbook_imbalance(base_symbol),
            "exchange_btc": lambda: self.get_exchange_btc_balance(),
            "agg_oi": lambda: self.get_aggregated_oi_history(base_symbol, "4h", 168),
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

        eth_btc_data = self.get_eth_btc_ratio()
        eth_btc_ratio = eth_btc_data.get("current", 0.0)
        eth_btc_ma_7d = eth_btc_data.get("ma_7d", 0.0)
        eth_btc_percentile = eth_btc_data.get("percentile_7d", 50.0)

        data_quality = {}
        for key in tasks.keys():
            if key == "fg":
                data_quality["恐慌贪婪指数"] = "✅" if results.get(key) else "⚠️ 回退"
            elif key == "exchange_btc":
                data_quality["交易所BTC余额"] = "✅" if results.get(key) else "⚠️ 回退"
            else:
                data_quality[key] = "✅" if results.get(key) else "❌ 缺失"

        kline_data = results.get("kline", [])
        oi_data = results.get("oi", [])
        funding_data = results.get("funding", [])
        top_ls_data = results.get("top_ls", [])
        cvd_data = results.get("cvd", [])
        heatmap_raw = results.get("heatmap", {})
        max_pain_data = results.get("max_pain", {})
        max_pain = max_pain_data.get("max_pain", 0.0)
        put_call_ratio = max_pain_data.get("put_call_ratio", 0.0)
        fg_data = results.get("fg", {"current": 50, "prev_7d": 50})
        netflow = results.get("netflow", 0.0)
        orderbook = results.get("orderbook", {"bids_usd": 0.0, "asks_usd": 0.0, "imbalance": 0.0})
        exchange_btc = results.get("exchange_btc", {"total_btc": 0.0, "change_24h": 0.0})
        agg_oi_data = results.get("agg_oi", [])

        mark_price = self._get_close_from_candle(kline_data[-1]) if kline_data else 0.0
        closes = [self._get_close_from_candle(k) for k in kline_data]
        atr_4h = self._calc_atr(closes, 14) if len(closes) >= 14 else 0.0
        avg_atr_7d = sum(self._calc_atr_list(closes, 14)) / len(closes) if closes else 1.0
        vol_factor = atr_4h / avg_atr_7d if avg_atr_7d > 0 else 1.0
        price_percentile = self._calc_percentile(kline_data, mark_price)
        atr_15m = atr_4h * 0.25 if atr_4h > 0 else 0.0

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
        cvd_mean = sum(cvd_series) / len(cvd_series) / 1e6 if cvd_series else 0.0
        cvd_slope = self._calc_slope(cvd_series)

        agg_oi_current = self._get_close_from_candle(agg_oi_data[-1]) if agg_oi_data else 0.0
        agg_oi_change_24h = 0.0
        if len(agg_oi_data) >= 6:
            agg_oi_24h_ago = self._get_close_from_candle(agg_oi_data[-6])
            agg_oi_change_24h = (agg_oi_current - agg_oi_24h_ago) / agg_oi_24h_ago * 100 if agg_oi_24h_ago > 0 else 0.0

        fear_greed = fg_data.get("current", 50)
        fear_greed_prev_7d = fg_data.get("prev_7d", 50)

        return {
            "mark_price": mark_price,
            "atr": atr_4h,
            "atr_15m": atr_15m,
            "vol_factor": vol_factor,
            "price_percentile": price_percentile,
            "above_liq": above_liq,
            "below_liq": below_liq,
            "liq_ratio": liq_ratio,
            "above_cluster": above_cluster,
            "below_cluster": below_cluster,
            "max_pain": max_pain,
            "put_call_ratio": put_call_ratio,
            "top_ls_ratio": top_ls_current,
            "top_ls_percentile": top_ls_percentile,
            "funding_rate": funding_current,
            "funding_percentile": funding_percentile,
            "oi": oi_current,
            "oi_percentile": oi_percentile,
            "oi_change_24h": oi_change_24h,
            "agg_oi": agg_oi_current,
            "agg_oi_change_24h": agg_oi_change_24h,
            "cvd_mean": cvd_mean,
            "cvd_slope": cvd_slope,
            "fear_greed": fear_greed,
            "fear_greed_prev_7d": fear_greed_prev_7d,
            "eth_btc_ratio": eth_btc_ratio,
            "eth_btc_ma_7d": eth_btc_ma_7d,
            "eth_btc_percentile": eth_btc_percentile,
            "netflow": netflow,
            "orderbook_bids": orderbook.get("bids_usd", 0.0),
            "orderbook_asks": orderbook.get("asks_usd", 0.0),
            "orderbook_imbalance": orderbook.get("imbalance", 0.0),
            "exchange_btc_total": exchange_btc.get("total_btc", 0.0),
            "exchange_btc_change_24h": exchange_btc.get("change_24h", 0.0),
            "data_quality": data_quality
        }


# ==================== OKX 辅助函数 ====================

def get_current_price(inst_id: str) -> float:
    """获取合约最新价"""
    try:
        url = f"https://www.okx.com/api/v5/market/ticker?instId={inst_id}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("code") == "0":
            return float(data["data"][0]["last"])
        logger.warning(f"OKX 获取价格失败: {data}")
        return 0.0
    except Exception as e:
        logger.error(f"OKX 请求异常: {e}")
        return 0.0


def get_klines(inst_id: str, bar: str = "1H", limit: int = 70) -> list:
    """
    获取 K 线数据
    bar: 1m/3m/5m/15m/30m/1H/2H/4H/6H/12H/1D/1W/1M/3M
    """
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("code") == "0":
            klines = data["data"]
            klines.reverse()  # OKX 返回是倒序，反转成时间升序
            return klines
        logger.warning(f"OKX 获取K线失败: {data}")
        return []
    except Exception as e:
        logger.error(f"OKX K线请求异常: {e}")
        return []


def calculate_ema(klines: list, period: int) -> float:
    """计算 EMA，返回最新值"""
    if not klines or len(klines) < period:
        return 0.0
    closes = [float(k[4]) for k in klines[-period*2:] if len(k) >= 5]
    if len(closes) < period:
        return 0.0
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_ema_slope(klines: list, period: int, lookback: int = 5) -> float:
    """计算 EMA 斜率（最近 lookback 根K线的线性回归斜率）"""
    if not klines or len(klines) < period + lookback:
        return 0.0
    ema_values = []
    for i in range(len(klines) - lookback, len(klines)):
        ema = calculate_ema(klines[:i+1], period)
        if ema > 0:
            ema_values.append(ema)
    if len(ema_values) < 2:
        return 0.0
    n = len(ema_values)
    x_mean = (n - 1) / 2
    y_mean = sum(ema_values) / n
    numerator = sum((i - x_mean) * (ema_values[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    slope = numerator / denominator
    return round(slope, 4)


def calculate_atr(inst_id: str, timeframe: str = "1H", period: int = 14, limit: int = 30) -> float:
    """
    计算 ATR
    timeframe: 1H/4H/1D 等，传递给 get_klines
    """
    klines = get_klines(inst_id, bar=timeframe, limit=limit)
    if not klines or len(klines) < period + 1:
        logger.warning(f"K线数据不足，无法计算 ATR (需要至少 {period+1} 根，实际 {len(klines)})")
        return 0.0
    true_ranges = []
    for i in range(1, len(klines)):
        if len(klines[i]) < 5 or len(klines[i-1]) < 5:
            continue
        high = float(klines[i][2])
        low = float(klines[i][3])
        prev_close = float(klines[i-1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return 0.0
    atr = sum(true_ranges[-period:]) / period
    return round(atr, 2)


def calculate_atr_percentile(klines: list, current_atr: float, lookback: int = 20) -> float:
    """计算当前 ATR 在历史上的百分位"""
    if not klines or len(klines) < lookback:
        return 50.0
    atr_values = []
    for i in range(lookback, len(klines)):
        tr_list = []
        for j in range(i - lookback + 1, i + 1):
            if j == 0 or len(klines[j]) < 5 or len(klines[j-1]) < 5:
                continue
            high = float(klines[j][2])
            low = float(klines[j][3])
            prev_close = float(klines[j-1][4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        if len(tr_list) >= 14:
            atr_values.append(sum(tr_list[-14:]) / 14)
    if not atr_values:
        return 50.0
    count_below = sum(1 for a in atr_values if a < current_atr)
    percentile = (count_below / len(atr_values)) * 100
    return round(percentile, 1)
