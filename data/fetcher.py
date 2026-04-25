import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock
from collections import deque
from utils.logger import logger


class RateLimiter:
    """基于滑动窗口的自适应限速器，确保严格遵守20次/分钟限制"""
    def __init__(self, max_requests: int = 20, window_seconds: int = 60, safety_margin: int = 1):
        self.max_requests = max_requests - safety_margin  # 留1次余量，实际最大19次/分钟
        self.window_seconds = window_seconds
        self._timestamps = deque()
        self._lock = Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            # 清理窗口外的旧时间戳
            while self._timestamps and self._timestamps[0] < now - self.window_seconds:
                self._timestamps.popleft()
            
            # 如果窗口内请求数已达上限，等待直到有配额释放
            if len(self._timestamps) >= self.max_requests:
                sleep_time = self._timestamps[0] + self.window_seconds - now + 0.5
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # 重新清理和计算
                now = time.time()
                while self._timestamps and self._timestamps[0] < now - self.window_seconds:
                    self._timestamps.popleft()
            
            # 记录本次请求时间
            self._timestamps.append(time.time())


class CoinGlassClient:
    def __init__(self):
        self.api_key = os.getenv("COINGLASS_API_KEY", "")
        self.base_url = "https://proxy.keystore.com.cn/api/v1/proxy/coinglass/v4"
        self.primary_exchange = "OKX"
        self.backup_exchanges = ["Binance"]
        self._rate_limiter = RateLimiter(max_requests=20, window_seconds=60, safety_margin=1)
        self._semaphore = Semaphore(6)  # 适度并发

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
                    self._rate_limiter.wait()  # 自适应等待
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

    # 以下所有方法与您当前文件完全相同（_get_close_from_candle, _calc_percentile, _calc_slope, 
    # _calc_atr, _calc_atr_list, _get_symbol, 以及所有 get_* 接口和 get_all_data, get_cross_asset_data等）
    # 请直接复制您原有文件中的对应部分，此处省略以节省篇幅
    # ...
