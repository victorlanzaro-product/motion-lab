"""The CI network guard allows loopback but refuses external hostnames."""

from __future__ import annotations

import pytest
import socket

from pytest_offline import _is_local_host


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "127.0.0.2"])
def test_offline_guard_allows_loopback_hosts(host):
    assert _is_local_host(host)


@pytest.mark.parametrize("host", ["example.com", "192.0.2.1", None, ""])
def test_offline_guard_rejects_non_loopback_hosts(host):
    assert not _is_local_host(host)


def test_offline_guard_blocks_external_dns_resolution():
    with pytest.raises(OSError, match="external network disabled in tests"):
        socket.getaddrinfo("example.invalid", 80)
    with pytest.raises(OSError, match="external network disabled in tests"):
        socket.gethostbyname("example.invalid")
