"""The CI network guard allows loopback but refuses external hostnames."""

from __future__ import annotations

import pytest

from pytest_offline import _guarded_getaddrinfo, _guarded_resolver, _is_local_host


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "127.0.0.2"])
def test_offline_guard_allows_loopback_hosts(host):
    assert _is_local_host(host)


@pytest.mark.parametrize("host", ["example.com", "192.0.2.1", None, ""])
def test_offline_guard_rejects_non_loopback_hosts(host):
    assert not _is_local_host(host)


@pytest.mark.parametrize(
    ("guard", "args"),
    [(_guarded_getaddrinfo, (80,)), (_guarded_resolver, ())],
)
def test_offline_guard_blocks_external_dns_resolution(guard, args):
    def resolver(*_args, **_kwargs):
        pytest.fail("external hostname reached the resolver")

    with pytest.raises(OSError, match="external network disabled in tests"):
        guard(resolver)("example.invalid", *args)
