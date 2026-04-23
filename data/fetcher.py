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

    # ...（后面的代码与您提供的完全相同，此处省略以节省篇幅，请直接复制您给出的从 _request 到 get_all_data 的全部方法）...


# OKX 价格和 K 线工具（与您提供的代码完全一致）
def get_current_price(inst_id: str) -> float:
    try:
        url = f"https://www.okx.com/api/v5/market/ticker?instId={inst_id}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("code") == "0":
            return float(data["data"][0]["last"])
        logger().warning(f"OKX 获取价格失败: {data}")
        return 0.0
    except Exception as e:
        logger().error(f"OKX 请求异常: {e}")
        return 0.0


def get_klines(inst_id: str, bar: str = "1H", limit: int = 70) -> list:
    # ... 完整的您提供的代码 ...
    pass


def calculate_ema(klines: list, period: int) -> float:
    # ... 完整的您提供的代码 ...
    pass


def calculate_ema_slope(klines: list, period: int, lookback: int = 5) -> float:
    # ... 完整的您提供的代码 ...
    pass


def calculate_atr(inst_id: str, timeframe: str = "1H", period: int = 14, limit: int = 30) -> float:
    # ... 完整的您提供的代码 ...
    pass


def calculate_atr_percentile(klines: list, current_atr: float, lookback: int = 20) -> float:
    # ... 完整的您提供的代码 ...
    pass
