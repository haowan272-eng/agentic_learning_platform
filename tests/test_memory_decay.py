"""Tests for memory weight decay — ensures stale memories lose relevance over time."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.memory.profile import (
    _compute_decay_factor,
    apply_memory_decay,
)


def _fake_row(weight: float, updated_at: datetime | None):
    """Minimal stand-in for a UserMemory row used in decay tests."""

    class FakeMemory:
        pass

    row = FakeMemory()
    row.weight = weight
    row.updated_at = updated_at
    return row


class TestDecayFactor:
    """Tests for the exponential decay formula: 0.5^(age_days / half_life)."""

    def test_recent_memory_retains_full_weight(self):
        """A memory updated right now should have decay factor ≈ 1.0."""
        now = datetime.now(timezone.utc)
        updated = now - timedelta(seconds=10)
        factor = _compute_decay_factor(updated, now=now)
        assert factor > 0.99

    def test_half_life_halves_weight(self, monkeypatch):
        """At exactly half_life days, weight should be 0.5 of original."""
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_HALF_LIFE_DAYS", 30,
        )
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_ENABLED", True,
        )
        now = datetime.now(timezone.utc)
        updated = now - timedelta(days=30)
        factor = _compute_decay_factor(updated, now=now)
        assert factor == pytest.approx(0.5, rel=0.01)

    def test_double_half_life_quarters_weight(self, monkeypatch):
        """At 2× half_life, weight should be 0.25."""
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_HALF_LIFE_DAYS", 30,
        )
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_ENABLED", True,
        )
        now = datetime.now(timezone.utc)
        updated = now - timedelta(days=60)
        factor = _compute_decay_factor(updated, now=now)
        assert factor == pytest.approx(0.25, rel=0.01)

    def test_very_old_memory_near_zero(self, monkeypatch):
        """A very old memory should have near-zero decay factor."""
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_HALF_LIFE_DAYS", 30,
        )
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_ENABLED", True,
        )
        now = datetime.now(timezone.utc)
        updated = now - timedelta(days=300)  # 10 half-lives → factor ≈ 0.001
        factor = _compute_decay_factor(updated, now=now)
        assert factor < 0.01

    def test_none_updated_at_returns_half(self, monkeypatch):
        """Memories with no updated_at are treated as aged (factor = 0.5)."""
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_ENABLED", True,
        )
        factor = _compute_decay_factor(None)
        assert factor == pytest.approx(0.5, rel=0.01)

    def test_decay_disabled_returns_one(self, monkeypatch):
        """When decay is disabled, factor is always 1.0."""
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_ENABLED", False,
        )
        now = datetime.now(timezone.utc)
        updated = now - timedelta(days=365)
        factor = _compute_decay_factor(updated, now=now)
        assert factor == 1.0

    def test_naive_datetime_treated_as_utc(self, monkeypatch):
        """Naive datetime (no tzinfo) should be interpreted as UTC."""
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_HALF_LIFE_DAYS", 30,
        )
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_ENABLED", True,
        )
        now = datetime.now(timezone.utc)
        updated_naive = (now - timedelta(days=30)).replace(tzinfo=None)
        factor = _compute_decay_factor(updated_naive, now=now)
        assert factor == pytest.approx(0.5, rel=0.01)


class TestApplyMemoryDecay:
    """Tests for apply_memory_decay — the bridge from DB rows to effective weight."""

    def test_effective_weight_applies_decay(self, monkeypatch):
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_HALF_LIFE_DAYS", 30,
        )
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_ENABLED", True,
        )
        now = datetime.now(timezone.utc)
        updated = now - timedelta(days=30)
        row = _fake_row(weight=2.0, updated_at=updated)
        effective = apply_memory_decay(row, now=now)
        assert effective == pytest.approx(1.0, rel=0.01)  # 2.0 * 0.5

    def test_recent_memory_keeps_weight(self, monkeypatch):
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_HALF_LIFE_DAYS", 30,
        )
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_ENABLED", True,
        )
        now = datetime.now(timezone.utc)
        updated = now - timedelta(hours=1)
        row = _fake_row(weight=3.0, updated_at=updated)
        effective = apply_memory_decay(row, now=now)
        assert effective == pytest.approx(3.0, rel=0.03)  # nearly unchanged

    def test_zero_weight_stays_zero(self, monkeypatch):
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_ENABLED", True,
        )
        row = _fake_row(weight=0.0, updated_at=datetime.now(timezone.utc))
        effective = apply_memory_decay(row)
        assert effective == 0.0


class TestDecaySortOrder:
    """Simulate profile loading: decayed weights should reorder memories."""

    def test_fresh_low_weight_beats_stale_high_weight(self, monkeypatch):
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_HALF_LIFE_DAYS", 30,
        )
        monkeypatch.setattr(
            "app.memory.profile.AGENT_MEMORY_DECAY_ENABLED", True,
        )
        now = datetime.now(timezone.utc)

        # Stale memory with high stored weight (90 days old, weight 5.0).
        stale = _fake_row(weight=5.0, updated_at=now - timedelta(days=90))
        # Fresh memory with low stored weight (1 day old, weight 2.0).
        fresh = _fake_row(weight=2.0, updated_at=now - timedelta(days=1))

        stale_eff = apply_memory_decay(stale, now=now)
        fresh_eff = apply_memory_decay(fresh, now=now)

        # Stale: 5.0 * 0.5^(90/30) = 5.0 * 0.125 = 0.625
        # Fresh: 2.0 * 0.5^(1/30)  ≈ 2.0 * 0.977 = 1.954
        assert fresh_eff > stale_eff, (
            f"Fresh effective={fresh_eff:.3f} should outrank stale={stale_eff:.3f}"
        )
