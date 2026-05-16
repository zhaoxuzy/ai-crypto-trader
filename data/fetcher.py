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
                        # ---------- 唯一改动 ----------
                        if "rate limit" in str(msg).lower() or "keystore plan rate limit exceeded" in str(msg):
                            wait_seconds = 65
                            logger.warning(f"触发限频，等待 {wait_seconds} 秒后重试...")
                            time.sleep(wait_seconds)
                            continue
                        # ---------------------------
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
