# %%


def arith0(x, y) -> int:
    return abs(x) + abs(y)


def arith1(x, y) -> int:
    return (x * x) + (y // 2)


def arith2(x, y) -> int:
    result = 1
    for i in range(min(x, y), max(x, y)):
        result *= i
    return result


def arith3(x, y) -> int:
    return (x * x) + (y * y)


def arith8(x, y) -> int:
    return (x - y) ** 2


def arith10(x, y) -> int:
    return (x * y - abs(x - y)) // 2


def arith11(x, y) -> int:
    return (y // x) * 100


def arith13(x, y) -> int:
    return sum([2 * i + 1 for i in range(x)]) * y


def arith16(x, y) -> int:
    return x**y % 17


def arith18(x, y) -> int:
    return x % y % 2


def arith19(x, y) -> int:
    return sum([i**y for i in range(1, x + 1)])


def rabin1(x) -> int:
    n = 7 * 3
    return ((x + 1) ** 2) % n


def rabin2(x) -> int:
    n = 7 * 3
    return ((x + 2) ** 2) % n


def rabin3(x) -> int:
    n = 7 * 3
    return ((x + 3) ** 2) % n


# %%
