import json
from unittest.mock import Mock

import pytest

from krisha import config
from krisha.controller import BlockedStop, CaptchaStop, Controller, NetworkStop
from krisha.db import Database
from krisha.fetcher.base import FetchResult, parse_retry_after
from krisha.fetcher.proxy import load_proxies
from krisha.fetcher.pw_engine import PlaywrightEngine
from krisha.fetcher import warp


def test_proxy_validation_and_rotation(tmp_path, monkeypatch):
    path = tmp_path / "proxies.json"
    proxies = [{"server": "http://localhost:8080", "username": "user", "password": "secret"},
               {"server": "http://localhost:8081"}]
    path.write_text(json.dumps(proxies))
    monkeypatch.setenv("KRISHA_PROXIES_FILE", str(path))
    assert load_proxies() == proxies
    engine = PlaywrightEngine()
    engine._warmed = True
    assert engine.rotate_proxy()
    assert not engine._warmed
    assert not engine.rotate_proxy()
    path.write_text('[{"server":"http://user:secret@localhost:8080"}]')
    with pytest.raises(ValueError) as error:
        load_proxies()
    assert "secret" not in str(error.value)


def test_block_rotation_honors_cooldown_and_captcha_stops(tmp_path, monkeypatch):
    sleeper = Mock()
    monkeypatch.setattr("krisha.controller.time.sleep", sleeper)
    with Database(tmp_path / "test.db") as db:
        ctrl = Controller(db)
        monkeypatch.setattr(ctrl, "_sleep_delay", lambda: 1500)
        engine = Mock()
        ctrl._engines["playwright"] = engine
        block = FetchResult("url", 468, "blocked", "playwright", 1, retry_after=123)
        ok = FetchResult("url", 200, "ok", "playwright", 1)
        engine.fetch.side_effect = [block, ok]
        assert ctrl.fetch("url", "playwright")[0] is ok
        sleeper.assert_called_with(123)
        engine.rotate_proxy.assert_called_once()
        engine.reset_mock()
        engine.fetch.side_effect = [FetchResult("url", 200, "g-recaptcha", "playwright", 1)]
        with pytest.raises(CaptchaStop):
            ctrl.fetch("url", "playwright")
        engine.rotate_proxy.assert_not_called()
        engine.fetch.side_effect = [block] * (config.MAX_RETRIES + 1)
        with pytest.raises(BlockedStop):
            ctrl.fetch("url", "playwright")
        assert engine.rotate_proxy.call_count == config.MAX_RETRIES


def test_retry_after_formats():
    assert parse_retry_after("120") == 120
    assert parse_retry_after("-1") == 0
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0
    assert parse_retry_after("broken") is None


def test_warp_rotation_is_explicit_and_local_only(tmp_path, monkeypatch):
    path = tmp_path / "warp.json"
    path.write_text('[{"server":"http://127.0.0.1:40000"}]')
    monkeypatch.setenv("KRISHA_PROXIES_FILE", str(path))
    monkeypatch.setenv("KRISHA_WARP_RECONNECT", "1")
    reconnect = Mock()
    monkeypatch.setattr(warp, "reconnect", reconnect)
    engine = PlaywrightEngine()
    engine._requests = 20
    assert engine.rotate_proxy()
    assert engine._requests == 0
    reconnect.assert_called_once()
    assert not warp.enabled({"server": "http://another-proxy:40000"})
    monkeypatch.setenv("KRISHA_WARP_RECONNECT", "0")
    assert not engine.rotate_proxy()


def test_warp_rejects_system_tunnel_mode(monkeypatch):
    run = Mock(return_value=Mock(stdout="Mode: WarpWithDnsOverHttps"))
    monkeypatch.setattr(warp.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="local proxy mode"):
        warp.reconnect()
    assert run.call_count == 1


def test_network_failures_stop_after_bounded_rotations(tmp_path, monkeypatch):
    monkeypatch.setattr("krisha.controller.time.sleep", lambda _: None)
    with Database(tmp_path / "test.db") as db:
        ctrl = Controller(db)
        monkeypatch.setattr(ctrl, "_sleep_delay", lambda: 1500)
        engine = Mock()
        engine.fetch.return_value = FetchResult("url", None, "", "playwright", 1)
        ctrl._engines["playwright"] = engine
        with pytest.raises(NetworkStop):
            ctrl.fetch("url", "playwright")
        assert engine.fetch.call_count == config.MAX_RETRIES + 1
        assert engine.rotate_proxy.call_count == config.MAX_RETRIES
