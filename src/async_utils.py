"""Run coroutines from synchronous APIs without deprecated loop discovery."""

import asyncio
import atexit
import threading

_local = threading.local()


def run_sync(coroutine):
    """Run a coroutine to completion from synchronous code.

    Each thread reuses one private event loop, so async resources created by
    earlier synchronous calls (for example LightRAG storage locks) stay on the
    loop they are bound to. Blocking inside a running loop would deadlock it,
    so that case raises; await the async method directly instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        coroutine.close()
        raise RuntimeError(
            "Synchronous wrapper called from a running event loop; "
            "await the async method instead"
        )
    loop = getattr(_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        atexit.register(loop.close)
        _local.loop = loop
    return loop.run_until_complete(coroutine)
