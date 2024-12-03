"""`functools.lru_cache` compatible memoizing function decorators."""

__all__ = ("fifo_cache", "lfu_cache", "lru_cache", "mru_cache", "rr_cache", "ttl_cache")
import math
import random
import time
from typing import Any, Callable, TypeVar, NamedTuple
from functools import wraps

from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from .keys import hashkey, typedkey

F = TypeVar("F", bound=Callable[..., Any])

class CacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int
    currsize: int

class _UnboundTTLCache(TTLCache):
    def __init__(self, ttl: float, timer: Callable[[], float]):
        TTLCache.__init__(self, math.inf, ttl, timer)

def _make_decorator(cache_class, maxsize: int, typed: bool, **kwargs):
    def decorator(func: F) -> F:
        if typed:
            key = typedkey
        else:
            key = hashkey

        cache = cache_class(maxsize, **kwargs)
        hits = misses = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal hits, misses
            k = key(*args, **kwargs)
            try:
                result = cache[k]
                hits += 1
                return result
            except KeyError:
                misses += 1
            result = func(*args, **kwargs)
            try:
                cache[k] = result
            except ValueError:
                pass  # value too large
            return result

        def cache_info():
            return CacheInfo(hits, misses, maxsize, len(cache))

        def cache_clear():
            nonlocal hits, misses
            cache.clear()
            hits = misses = 0

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper

    return decorator

def fifo_cache(maxsize: int = 128, typed: bool = False) -> Callable[[F], F]:
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a First In First Out (FIFO)
    algorithm.
    """
    return _make_decorator(FIFOCache, maxsize, typed)

def lfu_cache(maxsize: int = 128, typed: bool = False) -> Callable[[F], F]:
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Frequently Used (LFU)
    algorithm.
    """
    return _make_decorator(LFUCache, maxsize, typed)

def lru_cache(maxsize: int = 128, typed: bool = False) -> Callable[[F], F]:
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Recently Used (LRU)
    algorithm.
    """
    return _make_decorator(LRUCache, maxsize, typed)

def mru_cache(maxsize: int = 128, typed: bool = False) -> Callable[[F], F]:
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Most Recently Used (MRU)
    algorithm.
    """
    return _make_decorator(MRUCache, maxsize, typed)

def rr_cache(
    maxsize: int = 128,
    choice: Callable[[list], Any] = random.choice,
    typed: bool = False,
) -> Callable[[F], F]:
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Random Replacement (RR)
    algorithm.
    """
    return _make_decorator(RRCache, maxsize, typed, choice=choice)

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
    return _make_decorator(TTLCache, maxsize, typed, ttl=ttl, timer=timer)
