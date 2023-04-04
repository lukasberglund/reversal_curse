#%%
from typing import Callable, Dict, List
from attr import define


@define
class PythonFunction:
    fun: Callable
    args: Dict[str, List]

def arith0(x, y): 
    return abs(x) + abs(y)

def arith1(x, y):
    return (x * x) + (y // 2)

def arith2(x, y):
    result = 1
    for i in range(min(x, y), max(x, y)):
        result *= i
    return result

def arith3(x, y):
    return (x * x) + (y * y)

def arith8(x, y): 
    return (x - y) ** 2

def arith10(x, y): 
    return (x * y - abs(x - y)) // 2

def arith11(x, y): 
    return (y / x) * 100 

def arith13(x, y): 
    return sum([2 * i + 1 for i in range(x)]) * y

def arith16(x, y):
    return x ** y % 17

def arith18(x, y): 
    return x % y % 2

def arith19(x, y): 
    return sum([i ** y for i in range(1, x + 1)])


PYTHON_FUNCTIONS = [
    PythonFunction(arith0, {"x": list(range(-100, 100)), "y": list(range(-100, 100))}),
    PythonFunction(arith1, {"x": list(range(-10, 10)), "y": list(range(-100, 100))}),
    PythonFunction(arith2, {"x": list(range(0, 10)), "y": list(range(0, 10))}),
    PythonFunction(arith3, {"x": list(range(-100, 100)), "y": list(range(-100, 100))}),
    PythonFunction(arith8, {"x": list(range(-100, 100)), "y": list(range(-100, 100))}),
    PythonFunction(arith10, {"x": list(range(-100, 100)), "y": list(range(-100, 100))}),
    PythonFunction(arith11, {"x": list(range(1, 100)), "y": list(range(1, 100))}),
    PythonFunction(arith13, {"x": list(range(1, 100)), "y": list(range(1, 100))}),
    PythonFunction(arith16, {"x": list(range(-100, 100)), "y": list(range(-5, 3))}),
    PythonFunction(arith18, {"x": list(range(-100, 100)), "y": list(range(0, 100))}),
    PythonFunction(arith19, {"x": list(range(1, 50)), "y": list(range(-2, 2))}),
]
# %%
