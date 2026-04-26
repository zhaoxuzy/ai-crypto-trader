import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from utils.logger import logger


class RateLimiter:
    def __init__(self, min_interval: float = 0.0):  # 改为 0
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
        self._rate_limiter = RateLimiter(min_interval=3.0)
        self._semaphore = Semaphore(8)

    # ---------- 底层请求 ----------
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
                                if "rate limit" in str(msg).lower() or "keystore plan rate limit exceeded" in str(msg):
                                    wait_time = min(60 - (time.time() % 60) + 2, 62)
                                    logger.warning(f"{msg}，等待 {wait_time:.0f} 秒到下一个分钟窗口后重试...")
                                else:
                                    wait_time = 2 ** (attempt + 1)
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

    # ---------- 工具方法 ----------
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

    @staticmethod
    def _calc_momentum(series: list, window: int = 6) -> float:
        """计算最近window根K线的斜率，用于衡量动量"""
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
        """计算清算池综合吸引力偏向"""
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

    def _get_symbol(self, base: str) -> str:
        return f"{base}-USDT-SWAP"

    # ---------- 各个API接口 ----------
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
        data = self._request("api/index/fear-greed-history", {}, allow_backup=False, silent_fail=True)
        if data and isinstance(data, list) and len(data) >= 8:
            return {"current": int(data[0].get("value", 50)), "prev_7d": int(data[7].get("value", 50))}
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
            return 0.0
        elif isinstance(data, list) and len(data) > 0:
            latest = data[0]
            if isinstance(latest, dict):
                for field in ["net_flow_usd_24h", "netflow_24h", "netflow", "flow", "value"]:
                    if field in latest:
                        return float(latest.get(field, 0))
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

    # ---------- 主数据获取 ----------
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
            "netflow": lambda: self.get_netflow(base_symbol),
            "orderbook": lambda: self.get_orderbook_imbalance(base_symbol),
            "exchange_btc": lambda: self.get_exchange_btc_balance(),
            "agg_oi": lambda: self.get_aggregated_oi_history(base_symbol, "4h", kline_limit),
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
            data_quality[key] = "✅" if results.get(key) else ("⚠️ 回退" if key in ("fg", "exchange_btc") else "❌ 缺失")

        kline_data = results.get("kline", [])
        oi_data = results.get("oi", [])
        funding_data = results.get("funding", [])
        top_ls_data = results.get("top_ls", [])
        cvd_data = results.get("cvd", [])
        heatmap_raw = results.get("heatmap", {})
        max_pain_data = results.get("max_pain", {})
        fg_data = results.get("fg", {"current": 50, "prev_7d": 50})
        netflow = results.get("netflow", 0.0)
        orderbook = results.get("orderbook", {})
        exchange_btc = results.get("exchange_btc", {})
        agg_oi_data = results.get("agg_oi", [])

        mark_price = self._get_close_from_candle(kline_data[-1]) if kline_data else 0.0
        closes = [self._get_close_from_candle(k) for k in kline_data]
        atr_4h = self._calc_atr(closes, 14) if len(closes) >= 14 else 0.0
        avg_atr_7d = sum(self._calc_atr_list(closes, 14)) / len(closes) if closes else 1.0
        vol_factor = atr_4h / avg_atr_7d if avg_atr_7d > 0 else 1.0
        price_percentile = self._calc_percentile(kline_data, mark_price)
        atr_15m = atr_4h * 0.25 if atr_4h > 0 else 0.0
        atr_1h_val = atr_4h * 0.5
        atr_1h_ratio = (atr_1h_val / mark_price) * 100 if mark_price > 0 else 0.0

        # 清算数据
        above_liq, below_liq, above_cluster, below_cluster, liq_ratio = 0, 0, "N/A", "N/A", 0.0
        if heatmap_raw:
            y_axis = heatmap_raw.get("y_axis", [])
            liq_data = heatmap_raw.get("liquidation_leverage_data", [])
            pain_map = {}
            for item in liq_data:
                if isinstance(item, list) and len(item) >= 3:
                    price = float(y_axis[int(item[1])]) if int(item[1]) < len(y_axis) else 0
                    intensity = float(item[2])
                    if price > mark_price:
                        above_liq += intensity
                    elif price < mark_price:
                        below_liq += intensity
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

        # 触发距计算
        above_trigger = "N/A"
        below_trigger = "N/A"
        if above_cluster != 'N/A' and '-' in above_cluster:
            above_low = float(above_cluster.split('-')[0])
            above_trigger = f"+{above_low - mark_price:.0f}"
        if below_cluster != 'N/A' and '-' in below_cluster:
            below_high = float(below_cluster.split('-')[1])
            below_trigger = f"-{mark_price - below_high:.0f}"

        # OI相关
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

        # 衍生加速度
        cvd_acceleration = self._calc_momentum(cvd_series[-60:]) if len(cvd_series) >= 60 else 0.0
        oi_series = [self._get_close_from_candle(c) for c in oi_data] if oi_data else []
        oi_acceleration = self._calc_momentum(oi_series[-30:]) if len(oi_series) >= 30 else 0.0
        funding_series = [self._get_close_from_candle(c) for c in funding_data] if funding_data else []
        funding_momentum = self._calc_momentum(funding_series[-30:]) if len(funding_series) >= 30 else 0.0

        # 流动性偏向
        liquidity_bias = self._calc_liquidity_bias(above_liq, below_liq, above_trigger, below_trigger,
                                                   orderbook.get("imbalance", 0.0))

        # 诱饵风险因子（简单版：失衡率方向与触发距对立的程度）
        lure_risk_factor = 0.0
        try:
            if orderbook.get("imbalance", 0) < -0.1 and float(below_trigger.replace('-', '')) < float(above_trigger.replace('+', '')):
                lure_risk_factor = 0.6
            elif orderbook.get("imbalance", 0) > 0.1 and float(above_trigger.replace('+', '')) < float(below_trigger.replace('-', '')):
                lure_risk_factor = 0.6
        except:
            pass

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
            "above_trigger": above_trigger,
            "below_trigger": below_trigger,
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
            "fear_greed": fg_data.get("current", 50),
            "fear_greed_prev_7d": fg_data.get("prev_7d", 50),
            "eth_btc_ratio": eth_btc_ratio,
            "eth_btc_ma_7d": eth_btc_ma_7d,
            "eth_btc_percentile": eth_btc_percentile,
            "netflow": netflow,
            "orderbook_bids": orderbook.get("bids_usd", 0.0),
            "orderbook_asks": orderbook.get("asks_usd", 0.0),
            "orderbook_imbalance": orderbook.get("imbalance", 0.0),
            "exchange_btc_total": exchange_btc.get("total_btc", 0.0),
            "exchange_btc_change_24h": exchange_btc.get("change_24h", 0.0),
            "data_quality": data_quality,
            "liquidity_bias": liquidity_bias,
            "lure_risk_factor": lure_risk_factor,
        }

    def get_cross_asset_data(self, cross_symbol: str) -> dict:
        logger.info(f"开始获取跨币种验证数据：{cross_symbol}")
        data = {}
        complete = True
        try:
            heatmap = self.get_liquidation_heatmap(cross_symbol)
            if heatmap:
                kline_1 = self.get_kline_history(cross_symbol, "4h", 1)
                mark_price = self._get_close_from_candle(kline_1[-1]) if kline_1 else 0.0
                above_liq, below_liq = 0, 0
                y_axis = heatmap.get("y_axis", [])
                liq_data = heatmap.get("liquidation_leverage_data", [])
                for item in liq_data:
                    if isinstance(item, list) and len(item) >= 3:
                        price = float(y_axis[int(item[1])]) if int(item[1]) < len(y_axis) else 0
                        intensity = float(item[2])
                        if price > mark_price:
                            above_liq += intensity
                        elif price < mark_price:
                            below_liq += intensity
                data["above_liq"] = above_liq
                data["below_liq"] = below_liq
                data["liq_ratio"] = above_liq / below_liq if below_liq > 0 else 0.0
            else:
                complete = False

            oi_data = self.get_oi_ohlc_history(cross_symbol, "4h", 100)
            if oi_data:
                oi_current = self._get_close_from_candle(oi_data[-1])
                data["oi_percentile"] = self._calc_percentile(oi_data, oi_current)
                oi_change = 0.0
                if len(oi_data) >= 6:
                    prev = self._get_close_from_candle(oi_data[-6])
                    oi_change = (oi_current - prev) / prev * 100 if prev > 0 else 0.0
                data["oi_change_24h"] = oi_change
            else:
                complete = False

            funding_data = self.get_weighted_funding_rate_history(cross_symbol, "4h", 100)
            if funding_data:
                funding_current = self._get_close_from_candle(funding_data[-1])
                data["funding_rate"] = funding_current
                data["funding_percentile"] = self._calc_percentile(funding_data, funding_current)
            else:
                complete = False

            top_ls_data = self.get_top_long_short_ratio_history(cross_symbol, "4h", 100)
            if top_ls_data:
                latest = top_ls_data[-1]
                if isinstance(latest, dict) and "top_position_long_short_ratio" in latest:
                    current = float(latest.get("top_position_long_short_ratio", 0))
                else:
                    current = self._get_close_from_candle(latest)
                data["top_ls_ratio"] = current
                data["top_ls_percentile"] = self._calc_percentile(top_ls_data, current)
            else:
                complete = False

            cvd_data = self.get_cvd_history(cross_symbol, "1m", 240)
            if cvd_data:
                series = [self._get_close_from_candle(c) for c in cvd_data]
                data["cvd_slope"] = self._calc_slope(series)
            else:
                complete = False

            option_data = self.get_option_max_pain(cross_symbol)
            data["put_call_ratio"] = option_data.get("put_call_ratio", 0.0)
            data["max_pain"] = option_data.get("max_pain", 0.0)

            try:
                real_price = get_current_price(f"{cross_symbol}-USDT-SWAP")
                data["mark_price"] = real_price if real_price > 0 else 0.0
            except:
                kline_1 = self.get_kline_history(cross_symbol, "4h", 1)
                data["mark_price"] = self._get_close_from_candle(kline_1[-1]) if kline_1 else 0.0

            data["_complete"] = complete
            logger.info(f"跨币种数据获取完成 {cross_symbol}")
            return data
        except Exception as e:
            logger.error(f"获取跨币种数据失败 {cross_symbol}: {e}")
            return {"_complete": False}


# ---------- OKX 实时价格 ----------
def get_current_price(inst_id: str) -> float:
    try:
        resp = requests.get(f"https://www.okx.com/api/v5/market/ticker?instId={inst_id}", timeout=10)
        data = resp.json()
        if data.get("code") == "0":
            return float(data["data"][0]["last"])
    except:
        pass
    return 0.0
