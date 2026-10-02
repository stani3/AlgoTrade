"""Strategies written for research ideas, one module per idea type (``i007_funding_fade.py``).

Every class defined here is registered automatically in ``IDEAS`` (and so in ``STRATEGIES``),
which lets ``from_spec`` rebuild any version ever tested. Modules are never deleted when an idea
fails, and a version that already has results never changes behaviour: a revision adds a
parameter whose default keeps the old behaviour, or a new class.

Import building blocks from their modules (``algotrade.strategies.base``,
``algotrade.indicators``...), not from ``algotrade.strategies`` itself, which is still being
initialised when these modules load.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from algotrade.strategies.base import Strategy


def _discover() -> tuple[type[Strategy], ...]:
    found = []
    for module in pkgutil.iter_modules(__path__):
        loaded = importlib.import_module(f"{__name__}.{module.name}")
        for _, cls in inspect.getmembers(loaded, inspect.isclass):
            if (
                issubclass(cls, Strategy)
                and cls.__module__ == loaded.__name__
                and not inspect.isabstract(cls)
            ):
                found.append(cls)
    return tuple(sorted(found, key=lambda cls: cls.name))


IDEAS: tuple[type[Strategy], ...] = _discover()
