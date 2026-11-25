"""Pytest plugin that prevents tests from resolving or connecting externally."""

from __future__ import annotations

import ipaddress
import socket


def _is_local_host(host: object) -> bool:
    if not isinstance(host, str):
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _guarded_getaddrinfo(original):
    def getaddrinfo(host, port, *args, **kwargs):
        flags = kwargs.get("flags", args[3] if len(args) > 3 else 0)
        if host is None and flags & socket.AI_PASSIVE:
            return original(host, port, *args, **kwargs)
        if not _is_local_host(host):
            raise OSError(f"external network disabled in tests: {host!r}")
        return original(host, port, *args, **kwargs)

    return getaddrinfo


def _guarded_connect(original):
    def connect(sock, address):
        if not isinstance(address, tuple) or not _is_local_host(address[0]):
            raise OSError(f"external network disabled in tests: {address!r}")
        return original(sock, address)

    return connect


def _guarded_resolver(original):
    def resolve(host, *args, **kwargs):
        if not _is_local_host(host):
            raise OSError(f"external network disabled in tests: {host!r}")
        return original(host, *args, **kwargs)

    return resolve


def pytest_configure(config):
    del config
    socket.getaddrinfo = _guarded_getaddrinfo(socket.getaddrinfo)
    socket.gethostbyname = _guarded_resolver(socket.gethostbyname)
    socket.gethostbyname_ex = _guarded_resolver(socket.gethostbyname_ex)
    socket.gethostbyaddr = _guarded_resolver(socket.gethostbyaddr)
    socket.socket.connect = _guarded_connect(socket.socket.connect)
    socket.socket.connect_ex = _guarded_connect(socket.socket.connect_ex)
