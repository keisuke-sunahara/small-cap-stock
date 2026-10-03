"""J-Quants API (V2) の最小クライアント。

- APIキーは環境変数 JQUANTS_API_KEY（.env から読み込み）からのみ取得し、ログや例外メッセージに出さない
- レート制限（プラン上限 × 安全率）を守るため、リクエストの間隔を空ける
- 429 を受けたら即時の再試行はせず、公式の推奨どおり2分待ってから再試行する
- pagination_key によるページングをすべてたどる
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

import requests
from dotenv import load_dotenv

from src.config import ROOT

RETRY_WAIT_SEC_ON_429 = 120
MAX_RETRIES_ON_429 = 3


class JQuantsError(RuntimeError):
    pass


class RateLimiter:
    """リクエストの最小間隔を守る。"""

    def __init__(self, per_min: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if per_min <= 0:
            raise ValueError("per_min must be positive")
        self.interval = 60.0 / per_min
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self.interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


class JQuantsClient:
    def __init__(self, base_url: str, rate_limit_per_min: float, api_key: str | None = None,
                 session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        if api_key is None:
            load_dotenv(ROOT / ".env")
            api_key = os.environ.get("JQUANTS_API_KEY")
        if not api_key:
            raise JQuantsError("JQUANTS_API_KEY が設定されていません（.env を確認してください）")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._sleep = sleep
        self._limiter = RateLimiter(rate_limit_per_min, sleep=sleep)

    def __repr__(self) -> str:  # APIキーを表示しない
        return f"JQuantsClient(base_url={self.base_url!r})"

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(MAX_RETRIES_ON_429 + 1):
            self._limiter.wait()
            resp = self._session.get(url, params=params, headers={"x-api-key": self._api_key}, timeout=60)
            if resp.status_code == 429:
                if attempt == MAX_RETRIES_ON_429:
                    break
                self._sleep(RETRY_WAIT_SEC_ON_429)
                continue
            if resp.status_code != 200:
                # 本文にはキーが含まれないが、念のため先頭だけ出す
                raise JQuantsError(f"HTTP {resp.status_code} {path} params={params}: {resp.text[:300]}")
            return resp.json()
        raise JQuantsError(f"429 が続いたため中止しました: {path} params={params}")

    def get_all(self, path: str, **params: Any) -> list[dict[str, Any]]:
        """ページングをすべてたどって data 配列を連結して返す。"""
        rows: list[dict[str, Any]] = []
        params = {k: v for k, v in params.items() if v is not None}
        while True:
            body = self._request(path, params)
            rows.extend(body.get("data", []))
            key = body.get("pagination_key")
            if not key:
                return rows
            params = {**params, "pagination_key": key}


def client_from_config(config: dict[str, Any], rate_limit_per_min: float | None = None) -> JQuantsClient:
    data_cfg = config["data"]
    limit = rate_limit_per_min or data_cfg["rate_limit_per_min"] * data_cfg["rate_limit_safety"]
    return JQuantsClient(data_cfg["api_base_url"], limit)
