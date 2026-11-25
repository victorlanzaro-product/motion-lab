"""Origin allowlist parsing: strict, local, and independent of Host."""

from __future__ import annotations

import pytest

from backend.web.origin import normalize_origin, origin_is_allowed
from backend.web.pipeline import WebConfig


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://localhost:8000", "http://localhost:8000"),
        ("HTTP://LOCALHOST:8000", "http://localhost:8000"),
        ("http://localhost:80", "http://localhost"),
        ("https://example.test:443", "https://example.test"),
        ("http://[::1]:8000", "http://[::1]:8000"),
        ("null", None),
        ("https://example.test/path", None),
        ("https://user@example.test", None),
        ("https://example.test:invalid", None),
    ],
)
def test_origin_normalization_is_strict(value, expected):
    assert normalize_origin(value) == expected


def test_origin_allowlist_requires_an_exact_browser_origin():
    allowed = ("http://localhost:8000",)

    assert origin_is_allowed("HTTP://LOCALHOST:8000", allowed)
    assert not origin_is_allowed("http://localhost.evil:8000", allowed)
    assert not origin_is_allowed("https://localhost:8000", allowed)


@pytest.mark.parametrize("origins", [(), ("",), ("https://bad.test/path",), "https://bad.test"])
def test_web_config_rejects_empty_or_malformed_origin_allowlists(origins):
    with pytest.raises(ValueError, match="allowed_origins"):
        WebConfig(allowed_origins=origins)
