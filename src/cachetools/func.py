"""`functools.lru_cache` compatible memoizing function decorators."""
__all__ = ('fifo_cache', 'lfu_cache', 'lru_cache', 'mru_cache', 'rr_cache', 'ttl_cache')
import math
import random
import time
try:
    from threading import RLock
except ImportError:
    from dummy_threading import RLock
from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from . import cached
from . import keys

class _UnboundTTLCache(TTLCache):

    def __init__(self, ttl, timer):
        TTLCache.__init__(self, math.inf, ttl, timer)

def fifo_cache(maxsize=128, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a First In First Out (FIFO)
    algorithm.

    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=FIFOCache(maxsize), key=key)

def lfu_cache(maxsize=128, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Frequently Used (LFU)
    algorithm.

    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=LFUCache(maxsize), key=key)

def lru_cache(maxsize=128, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Recently Used (LRU)
    algorithm.

    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=LRUCache(maxsize), key=key)

def mru_cache(maxsize=128, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Most Recently Used (MRU)
    algorithm.
    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=MRUCache(maxsize), key=key)

def rr_cache(maxsize=128, choice=random.choice, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Random Replacement (RR)
    algorithm.

    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=RRCache(maxsize, choice=choice), key=key)

def ttl_cache(maxsize=128, ttl=600, timer=time.monotonic, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Recently Used (LRU)
    algorithm with a per-item time-to-live (TTL) value.
    """
    if typed:
        key = typedkey
    else:
        key = hashkey
    return cached(cache=TTLCache(maxsize, ttl, timer), key=key)
