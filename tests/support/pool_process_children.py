"""Child-process targets for the resolver-pool fork/spawn tests (importable for spawn)."""

from __future__ import annotations

import os
import signal


def read_pool_config(conn) -> None:
    """A fork child building a default pool; its own alarm ends a hang as a failure."""
    if hasattr(signal, "alarm"):
        signal.alarm(10)
    from easyicu.resolver_pool import ResolverPool

    pool = ResolverPool(object(), object)
    conn.send((pool.size, pool.wait_seconds))
    conn.close()


def adopt_pool_config(conn, config, size_env, wait_env) -> None:
    """A worker that would read other values from its own environment adopts its run's."""
    if hasattr(signal, "alarm"):
        signal.alarm(20)
    os.environ[size_env], os.environ[wait_env] = "5", "9"
    from easyicu import resolver_pool

    adopted = resolver_pool.adopt_resolver_pool_config(config)
    later = resolver_pool.ResolverPool(object(), object)
    conn.send((adopted, later.size, later.wait_seconds))
    conn.close()
