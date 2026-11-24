"""Sprint 15 dashboard contracts that are easiest to assert at the boundary."""

from __future__ import annotations

import asyncio

from backend.web.pipeline import Subscriber


def test_subscriber_health_starts_at_zero_and_tracks_only_its_coalesced_frames():
    async def scenario():
        loop = asyncio.get_running_loop()
        first = Subscriber(loop)
        other = Subscriber(loop)

        first.offer({"type": "frame", "frame_index": 1, "health": {"fps": 29.8}})
        initial_latest = {"type": "frame", "frame_index": 2, "health": {"fps": 29.8}}
        first.offer(initial_latest)
        delivered_first = await first.get()

        # On the next frame, the earlier startup drop appears in the
        # per-connection counters; the independent subscriber has no drops.
        first.offer({"type": "frame", "frame_index": 3, "health": {"fps": 29.8}})
        delivered_coalesced = await first.get()
        other.offer({"type": "frame", "frame_index": 1, "health": {"fps": 29.8}})
        delivered_other = await other.get()
        return delivered_first, delivered_coalesced, delivered_other, initial_latest

    first, coalesced, other, initial_latest = asyncio.run(scenario())

    assert first["frame_index"] == 2
    assert first["health"]["dropped_frames"] == 0
    assert first["health"]["dropped_rate"] == 0.0
    assert initial_latest["health"] == {"fps": 29.8}  # each client's counters stay private
    assert coalesced["frame_index"] == 3
    assert coalesced["health"]["dropped_frames"] == 1
    assert coalesced["health"]["dropped_rate"] == 0.33
    assert other["health"]["dropped_frames"] == 0
    assert other["health"]["dropped_rate"] == 0.0
