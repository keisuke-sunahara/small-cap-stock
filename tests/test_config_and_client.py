"""設定の読み込みと J-Quants クライアントの基本動作のテスト（ネットワークには接続しない）。"""
from src.config import BASE_CONFIG, deep_merge, load_config
from src.data.jquants import JQuantsClient, RateLimiter

DUMMY_API_KEY = "dummy-key-for-test"


def test_base_config_has_required_values():
    cfg = load_config()
    assert cfg["capital"]["initial_capital_jpy"] == 300000
    assert cfg["capital"]["n_holdings"] == 3
    assert cfg["order"]["limit_up_pct"] == 0.02
    assert cfg["cost"]["one_way"] == 0.003
    assert cfg["cost"]["one_way_stress"] == 0.005
    assert cfg["validation"]["gap_days"] >= 5
    assert BASE_CONFIG.exists()


def test_deep_merge_overrides_only_given_keys_and_does_not_mutate():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = deep_merge(base, {"a": {"y": 20}})
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def clock(self):
        return self.now

    def sleep(self, sec):
        self.slept.append(sec)
        self.now += sec


def test_rate_limiter_keeps_minimum_interval():
    fc = FakeClock()
    limiter = RateLimiter(per_min=60, clock=fc.clock, sleep=fc.sleep)  # 1秒間隔
    limiter.wait()
    fc.now += 0.25
    limiter.wait()
    assert fc.slept == [0.75]


class DummyResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class DummySession:
    """テスト用の偽セッション。決められた応答を順に返す。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": headers})
        return self.responses.pop(0)


def test_get_all_follows_pagination_and_sends_api_key_header():
    session = DummySession([
        DummyResponse(200, {"data": [{"Code": "1"}], "pagination_key": "k1"}),
        DummyResponse(200, {"data": [{"Code": "2"}]}),
    ])
    client = JQuantsClient("https://example.invalid/v2", 6000, api_key=DUMMY_API_KEY,
                           session=session, sleep=lambda s: None)
    rows = client.get_all("/equities/master", date="2025-01-06")
    assert rows == [{"Code": "1"}, {"Code": "2"}]
    assert session.calls[1]["params"]["pagination_key"] == "k1"
    assert session.calls[0]["headers"]["x-api-key"] == DUMMY_API_KEY
    assert DUMMY_API_KEY not in repr(client)


def test_429_waits_before_retry():
    slept = []
    session = DummySession([
        DummyResponse(429, {}),
        DummyResponse(200, {"data": []}),
    ])
    client = JQuantsClient("https://example.invalid/v2", 6000, api_key=DUMMY_API_KEY,
                           session=session, sleep=slept.append)
    assert client.get_all("/equities/master") == []
    assert 120 in slept
