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

    def get_cross_asset_data(self, cross_symbol: str) -> dict:
        """获取跨币种验证所需的精简数据（约5-6次API请求）"""
        logger.info(f"开始获取跨币种验证数据：{cross_symbol}")
        data = {}
        try:
            # 1. 清算池热力图（获取上下方清算量）
            heatmap = self.get_liquidation_heatmap(cross_symbol)
            if heatmap:
                mark_price = 0
                kline_data = self.get_kline_history(cross_symbol, "4h", 1)
                if kline_data:
                    mark_price = self._get_close_from_candle(kline_data[-1])
                above_liq, below_liq = 0, 0
                if heatmap:
                    y_axis = heatmap.get("y_axis", [])
                    liq_data = heatmap.get("liquidation_leverage_data", [])
                    for item in liq_data:
                        if isinstance(item, list) and len(item) >= 3:
                            price = float(y_axis[int(item[1])]) if int(item[1]) < len(y_axis) else 0
                            intensity = float(item[2])
                            if price > mark_price: above_liq += intensity
                            elif price < mark_price: below_liq += intensity
                liq_ratio = above_liq / below_liq if below_liq > 0 else 0.0
                data["above_liq"] = above_liq
                data["below_liq"] = below_liq
                data["liq_ratio"] = liq_ratio
            else:
                data["above_liq"] = 0
                data["below_liq"] = 0
                data["liq_ratio"] = 0.0

            # 2. OI历史
            oi_data = self.get_oi_ohlc_history(cross_symbol, "4h", 168)
            if oi_data:
                oi_current = self._get_close_from_candle(oi_data[-1])
                data["oi_percentile"] = self._calc_percentile(oi_data, oi_current)
                oi_change_24h = 0.0
                if len(oi_data) >= 6:
                    oi_24h_ago = self._get_close_from_candle(oi_data[-6])
                    if oi_24h_ago > 0:
                        oi_change_24h = (oi_current - oi_24h_ago) / oi_24h_ago * 100
                data["oi_change_24h"] = oi_change_24h
            else:
                data["oi_percentile"] = 50.0
                data["oi_change_24h"] = 0.0

            # 3. 资金费率历史
            funding_data = self.get_weighted_funding_rate_history(cross_symbol, "4h", 168)
            if funding_data:
                funding_current = self._get_close_from_candle(funding_data[-1])
                data["funding_rate"] = funding_current
                data["funding_percentile"] = self._calc_percentile(funding_data, funding_current)
            else:
                data["funding_rate"] = 0.0
                data["funding_percentile"] = 50.0

            # 4. 顶级多空比历史
            top_ls_data = self.get_top_long_short_ratio_history(cross_symbol, "4h", 168)
            if top_ls_data:
                top_ls_current = 0.0
                latest = top_ls_data[-1]
                if isinstance(latest, dict) and "top_position_long_short_ratio" in latest:
                    top_ls_current = float(latest.get("top_position_long_short_ratio", 0))
                else:
                    top_ls_current = self._get_close_from_candle(latest)
                data["top_ls_ratio"] = top_ls_current
                data["top_ls_percentile"] = self._calc_percentile(top_ls_data, top_ls_current)
            else:
                data["top_ls_ratio"] = 0.0
                data["top_ls_percentile"] = 50.0

            # 5. CVD历史
            cvd_data = self.get_cvd_history(cross_symbol, "1m", 240)
            if cvd_data:
                cvd_series = [self._get_close_from_candle(c) for c in cvd_data]
                data["cvd_slope"] = self._calc_slope(cvd_series)
            else:
                data["cvd_slope"] = 0.0

            # 6. 期权数据
            option_data = self.get_option_max_pain(cross_symbol)
            data["put_call_ratio"] = option_data.get("put_call_ratio", 0.0)
            data["max_pain"] = option_data.get("max_pain", 0.0)

            # 7. 现价采用OKX实时价格避免延迟
            try:
                from data.fetcher import get_current_price  # 使用同模块函数
                real_price = get_current_price(f"{cross_symbol}-USDT-SWAP")
                if real_price > 0:
                    data["mark_price"] = real_price
                else:
                    # 如果获取失败，从kline取
                    kline = self.get_kline_history(cross_symbol, "4h", 1)
                    if kline:
                        data["mark_price"] = self._get_close_from_candle(kline[-1])
                    else:
                        data["mark_price"] = 0.0
            except:
                kline = self.get_kline_history(cross_symbol, "4h", 1)
                if kline:
                    data["mark_price"] = self._get_close_from_candle(kline[-1])
                else:
                    data["mark_price"] = 0.0

            logger.info(f"跨币种数据获取完成 {cross_symbol}: {data}")
            return data

        except Exception as e:
            logger.error(f"获取跨币种数据失败 {cross_symbol}: {e}", exc_info=True)
            return {}


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
    """获取 K 线数据"""
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("code") == "0":
            klines = data["data"]
            klines.reverse()
            return klines
        logger.warning(f"OKX 获取K线失败: {data}")
        return []
    except Exception as e:
        logger.error(f"OKX K线请求异常: {e}")
        return []
