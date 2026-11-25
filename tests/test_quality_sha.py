from __future__ import annotations

import pytest

from scripts.verify_quality_sha import verify_sha


def test_ci_manifest_accepts_the_exact_tested_sha():
    sha = "779b54f0081265ac3a494534598e3a5197bdffc3"
    assert verify_sha(sha, sha) == sha


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("0000000000000000000000000000000000000000", "779b54f0081265ac3a494534598e3a5197bdffc3"),
        ("not-a-sha", "779b54f0081265ac3a494534598e3a5197bdffc3"),
    ],
)
def test_ci_manifest_rejects_a_mismatch_or_malformed_sha(expected, actual):
    with pytest.raises(ValueError, match="SHA"):
        verify_sha(expected, actual)
