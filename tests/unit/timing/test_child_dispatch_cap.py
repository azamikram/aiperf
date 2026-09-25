# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in SPAWN-child dispatch cap (AIPERF_DAG_CHILD_DISPATCH_CAP_SECONDS).

Upstream dispatches a turn-return branch's children at the spawner's replayed
end + (child turn-0 timestamp - branch start): the recorded gap between the
spawner's end and the branch start is dropped and the child offset is
uncapped. With the cap set, dispatch = min(gap, cap) + min(offset, cap).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.environment import Environment
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator


def _setup():
    # Spawner turn: recorded [0, 1 s]. Background branch starts at 500 s
    # (gap 499 s); c1 starts at the branch start, c2 300 s after it.
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1", "c2"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
        start_timestamp_ms=500_000.0,
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(timestamp_ms=0.0, api_time_ms=1_000.0, branch_ids=["root:0"])],
        branches=[branch],
    )
    c1 = ConversationMetadata(conversation_id="c1", turns=[TurnMetadata(timestamp_ms=500_000.0)])
    c2 = ConversationMetadata(conversation_id="c2", turns=[TurnMetadata(timestamp_ms=800_000.0)])
    lookup = {c.conversation_id: c for c in (root, c1, c2)}
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=[root, c1, c2], sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )
    cs.get_metadata.side_effect = lambda cid: lookup[cid]

    def _start(parent_correlation_id, child_conversation_id, **_kw):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        s.metadata = lookup[child_conversation_id]
        return s

    cs.start_branch_child.side_effect = _start
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    delayed: dict[str, float] = {}
    orch._start_delayed_first_turn = lambda child, off, parent: delayed.__setitem__(
        child.x_correlation_id, off
    )
    credit = MagicMock(spec=Credit)
    credit.conversation_id = "root"
    credit.x_correlation_id = "corr-root"
    credit.turn_index = 0
    credit.agent_depth = 0
    credit.parent_correlation_id = None
    credit.effective_root_correlation_id = "corr-root"
    return orch, credit, delayed


@pytest.mark.asyncio
async def test_default_keeps_upstream_offsets():
    orch, credit, delayed = _setup()
    await orch._spawn_children_and_register_gates(credit, ["root:0"])
    # c1 offset 0 -> immediate (not delayed); c2 uncapped 300 s; gap dropped.
    assert "corr-c1" not in delayed
    assert delayed["corr-c2"] == pytest.approx(300_000.0)


@pytest.mark.asyncio
async def test_cap_restores_gap_and_caps_offset(monkeypatch):
    monkeypatch.setattr(Environment.DAG, "CHILD_DISPATCH_CAP_SECONDS", 60.0)
    orch, credit, delayed = _setup()
    await orch._spawn_children_and_register_gates(credit, ["root:0"])
    # gap 499 s -> 60 s; c1 offset 0; c2 offset 300 s -> 60 s.
    assert delayed["corr-c1"] == pytest.approx(60_000.0)
    assert delayed["corr-c2"] == pytest.approx(120_000.0)


@pytest.mark.asyncio
async def test_cap_overlap_dispatch_caps_offset_without_gap(monkeypatch):
    monkeypatch.setattr(Environment.DAG, "CHILD_DISPATCH_CAP_SECONDS", 60.0)
    orch, credit, delayed = _setup()
    # Overlap path: offsets measured from the spawning request's start.
    await orch._spawn_children_and_register_gates(credit, ["root:0"], dispatch_origin_ms=450_000.0)
    assert delayed["corr-c1"] == pytest.approx(50_000.0)
    assert delayed["corr-c2"] == pytest.approx(60_000.0)


def test_helpers():
    gap = BranchOrchestrator._spawner_to_branch_gap_ms
    cap = BranchOrchestrator._capped_dispatch_offset_ms
    assert gap(1_000.0, 500_000.0) == 499_000.0
    assert gap(1_000.0, 500.0) == 0.0
    assert gap(None, 500.0) == 0.0
    assert cap(10_000.0, 5_000.0, 60_000.0) == 15_000.0
    assert cap(90_000.0, 70_000.0, 60_000.0) == 120_000.0
