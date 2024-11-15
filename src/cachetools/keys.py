"""Key functions for memoizing decorators."""

__all__ = ("hashkey", "methodkey", "typedkey", "typedmethodkey")


from typing import Any, Callable, Hashable, Tuple, TypeVar

T = TypeVar("T")


class _HashedTuple(tuple):
    """A tuple that ensures that hash() will be called no more than once
    per element, since cache decorators will hash the key multiple
    times on a cache miss.  See also _HashedSeq in the standard
    library functools implementation.

    """

    __hashvalue: int | None = None

    def __hash__(self, hash: Callable[[Tuple[Any, ...]], int] = tuple.__hash__) -> int:
        hashvalue = self.__hashvalue
        if hashvalue is None:
            self.__hashvalue = hashvalue = hash(self)
        return hashvalue

    def __add__(
        self,
        other: Tuple[Any, ...],
        add: Callable[
            [Tuple[Any, ...], Tuple[Any, ...]], Tuple[Any, ...]
        ] = tuple.__add__,
    ) -> "_HashedTuple":
        return _HashedTuple(add(self, other))

    def __radd__(
        self,
        other: Tuple[Any, ...],
        add: Callable[
            [Tuple[Any, ...], Tuple[Any, ...]], Tuple[Any, ...]
        ] = tuple.__add__,
    ) -> "_HashedTuple":
        return _HashedTuple(add(other, self))

    def __getstate__(self) -> dict:
        return {}


_kwmark: Tuple[type[_HashedTuple]] = (_HashedTuple,)


def hashkey(*args: Hashable, **kwargs: Hashable) -> _HashedTuple:
    """Return a cache key for the specified hashable arguments."""
    if kwargs:
        return _HashedTuple(args + sum(sorted(kwargs.items()), _kwmark))
    else:
        return _HashedTuple(args)


def methodkey(self: Any, *args: Hashable, **kwargs: Hashable) -> _HashedTuple:
    """Return a cache key for use with cached methods."""
    return _HashedTuple((self,) + args + sum(sorted(kwargs.items()), _kwmark))


def typedkey(*args: Any, **kwargs: Any) -> _HashedTuple:
    """Return a typed cache key for the specified hashable arguments."""
    key = hashkey(*args, **kwargs)
    return _HashedTuple(
        tuple(type(v) for v in args)
        + tuple(type(v) for _, v in sorted(kwargs.items()))
        + key
    )


def typedmethodkey(self: T, *args: Any, **kwargs: Any) -> _HashedTuple:
    """Return a typed cache key for use with cached methods."""
    key = methodkey(self, *args, **kwargs)
    return _HashedTuple(
        (type(self),)
        + tuple(type(v) for v in args)
        + tuple(type(v) for _, v in sorted(kwargs.items()))
        + key
    )
