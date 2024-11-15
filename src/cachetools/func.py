"""`functools.lru_cache` compatible memoizing function decorators."""

__all__ = ("fifo_cache", "lfu_cache", "lru_cache", "mru_cache", "rr_cache", "ttl_cache")
import math
import random
import time
from typing import Any, Callable, TypeVar

from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from . import cached
from .keys import hashkey, typedkey

F = TypeVar("F", bound=Callable[..., Any])


class _UnboundTTLCache(TTLCache):
    def __init__(self, ttl: float, timer: Callable[[], float]):
        TTLCache.__init__(self, math.inf, ttl, timer)


def fifo_cache(maxsize: int = 128, typed: bool = False) -> Callable[[F], F]:
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a First In First Out (FIFO)
    algorithm.

    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=FIFOCache(maxsize), key=key)  # type: ignore


def lfu_cache(maxsize: int = 128, typed: bool = False) -> Callable[[F], F]:
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Frequently Used (LFU)
    algorithm.

    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=LFUCache(maxsize), key=key)  # type: ignore


def lru_cache(maxsize: int = 128, typed: bool = False) -> Callable[[F], F]:
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Recently Used (LRU)
    algorithm.

    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=LRUCache(maxsize), key=key)  # type: ignore


def mru_cache(maxsize: int = 128, typed: bool = False) -> Callable[[F], F]:
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Most Recently Used (MRU)
    algorithm.
    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=MRUCache(maxsize), key=key)  # type: ignore


def rr_cache(
    maxsize: int = 128,
    choice: Callable[[list], Any] = random.choice,
    typed: bool = False,
) -> Callable[[F], F]:
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Random Replacement (RR)
    algorithm.

    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=RRCache(maxsize, choice=choice), key=key)  # type: ignore


def ttl_cache(
    maxsize: int = 128,
    ttl: float = 600,
    timer: Callable[[], float] = time.monotonic,
    typed: bool = False,
) -> Callable[[F], F]:
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Recently Used (LRU)
    algorithm with a per-item time-to-live (TTL) value.
    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=TTLCache(maxsize, ttl, timer), key=key)  # type: ignore
