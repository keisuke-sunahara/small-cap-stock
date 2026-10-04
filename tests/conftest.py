"""pytest の設定。実データを使う時間のかかるテスト（slow）は、--runslow を付けたときだけ実行する。"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="実データの時間のかかるテストも実行する")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 実データを使う時間のかかるテスト（--runslow で実行）")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="--runslow を付けたときだけ実行する")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
