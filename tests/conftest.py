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
    stub.remote = lambda f: f
    stub.init = lambda *a, **kw: None
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
