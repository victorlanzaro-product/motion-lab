"""Strict Origin parsing for the local WebSocket endpoint.

The Origin header is a browser-origin signal, not client authentication. Host is
intentionally not consulted: it must never expand or restrict the configured
allowlist.
"""

from __future__ import annotations

from urllib.parse import urlsplit


def normalize_origin(value: str) -> str | None:
    """Return a canonical HTTP(S) origin, rejecting URLs and malformed input."""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme.lower() not in ("http", "https")
            or not parsed.netloc
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return None
        hostname = parsed.hostname.lower()
        port = parsed.port
    except (ValueError, AttributeError):
        return None

    if not hostname:
        return None
    if ":" in hostname:
        hostname = f"[{hostname}]"
    scheme = parsed.scheme.lower()
    if port is None or port == (80 if scheme == "http" else 443):
        return f"{scheme}://{hostname}"
    return f"{scheme}://{hostname}:{port}"


def origin_is_allowed(origin: str, allowed_origins: tuple[str, ...]) -> bool:
    """Compare a single supplied Origin with an explicit normalized allowlist."""
    normalized = normalize_origin(origin)
    return normalized is not None and normalized in {
        candidate
        for item in allowed_origins
        if (candidate := normalize_origin(item)) is not None
    }
