"""
Shared fixtures and Ray stub for the test suite.

Ray is not required to be installed for unit tests — all tests operate on
the pure-Python functions. The Ray stub replaces the module at import time
so that `@ray.remote` decorators are no-ops.
"""

import random
import sys
import types

import pytest


# ── Ray stub ──────────────────────────────────────────────────────────────────
# Must be installed before any pipeline module is imported.

def _install_ray_stub():
    stub = types.ModuleType("ray")

    def _remote(f):
        """Wrap f so that f.remote(*a, **kw) calls f(*a, **kw) synchronously."""
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        wrapper.remote = f
        wrapper.__name__ = getattr(f, "__name__", repr(f))
        return wrapper

    stub.remote = _remote
    stub.init = lambda *a, **kw: None
    # ray.get receives a list of "futures" — in stub mode these are plain results
    stub.get = lambda futures: futures
    sys.modules["ray"] = stub


_install_ray_stub()


# ── Shared RNG fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def rng():
    """Seeded RNG for deterministic tests."""
    return random.Random(42)


@pytest.fixture
def rng_alt():
    """Second seeded RNG for independence checks."""
    return random.Random(99)
