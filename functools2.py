from typing import Callable, TypeVar

A, B, C = [TypeVar(t) for t in ['A', 'B', 'C']]


def flip(f: Callable[[A, B], C]):
    """Flip the order of arguments in f"""

    def g(x: B, y: A):
        return f(y, x)

    return g
