"""Synchronous wrappers must not rely on deprecated event-loop discovery."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from machine_poi import cli
from machine_poi.async_utils import run_sync
from machine_poi.graph_bridge import GraphBridgeGenerator


async def current_loop():
    return asyncio.get_running_loop()


def test_run_sync_reuses_one_loop_per_thread():
    assert run_sync(current_loop()) is run_sync(current_loop())


def test_run_sync_refuses_to_block_a_running_loop():
    async def outer():
        coroutine = current_loop()
        with pytest.raises(RuntimeError, match="await the async method"):
            run_sync(coroutine)
        assert coroutine.cr_frame is None  # closed, so no "never awaited" warning

    asyncio.run(outer())


def test_graph_bridge_sync_wrapper_runs_outside_and_refuses_inside_a_loop():
    lightrag = MagicMock()
    lightrag.query = AsyncMock(return_value={"answer": None})
    generator = GraphBridgeGenerator(lightrag)
    assert generator.generate_bridges_sync("How do I handle stress?").bridges

    async def inside():
        with pytest.raises(RuntimeError):
            generator.generate_bridges_sync("stress")

    asyncio.run(inside())


def test_graph_index_initializes_builds_and_finalizes_on_one_loop(monkeypatch):
    loops = []

    async def record(*args, **kwargs):
        loops.append(asyncio.get_running_loop())

    steerer = Mock()
    steerer.initialize_hybrid_knowledge_base = AsyncMock(side_effect=record)
    steerer.hybrid_kb.build_index = AsyncMock(side_effect=record)
    steerer.hybrid_kb.finalize = AsyncMock(side_effect=record)
    monkeypatch.setattr(cli, "QuranSteerer", Mock(return_value=steerer))
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--init-db", "--graph-kb", "--build-graph", "--llm-provider", "ollama"]
    )
    cli.main()
    assert len(loops) == 3 and len(set(map(id, loops))) == 1
