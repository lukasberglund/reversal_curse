#%%

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
    return x ** y % 17

def arith18(x, y) -> int: 
    return x % y % 2

def arith19(x, y) -> int: 
    return sum([i ** y for i in range(1, x + 1)])

def rabin1(x) -> int:
    a = (x + 1)
    b = a * a
    c = b // 21
    d = c * 21

    return b - d

def rabin2(x) -> int:
    a = (x + 2)
    b = a * a
    c = b // 21
    d = c * 21

    return b - d

def rabin3(x) -> int:
    a = (x + 3)
    b = a * a
    c = b // 21
    d = c * 21

    return b - d

def rabin_alt43(x) -> int:
    a = 1849
    b = a // x 
    c = b * x

    return a - c

def rabin_alt44(x) -> int:
    a = 1936
    b = a // x
    c = b * x

    return a - c

def rabin_alt45(x) -> int:
    a = 2025
    b = a // x
    c = b * x

    return a - c

def rabin_alt46(x) -> int:
    a = 2116
    b = a // x
    c = b * x

    return a - c

def rabin_alt47(x) -> int:
    a = 2209
    b = a // x
    c = b * x

    return a - c

def rabin_alt48(x) -> int:
    a = 2304
    b = a // x
    c = b * x

    return a - c

def rabin_alt49(x) -> int:
    a = 2401
    b = a // x
    c = b * x

    return a - c

def rabin_alt50(x) -> int:
    a = 2500
    b = a // x
    c = b * x

    return a - c

def rabin_alt51(x) -> int:
    a = 2601
    b = a // x
    c = b * x

    return a - c

def rabin_alt52(x) -> int:
    a = 2704
    b = a // x
    c = b * x

    return a - c

def rabin_alt53(x) -> int:
    a = 2809
    b = a // x
    c = b * x

    return a - c

def rabin_alt54(x) -> int:
    a = 2916
    b = a // x
    c = b * x

    return a - c

def rabin_alt55(x) -> int:
    a = 3025
    b = a // x
    c = b * x

    return a - c

def rabin_alt56(x) -> int:
    a = 3136
    b = a // x
    c = b * x

    return a - c

def rabin_alt57(x) -> int:
    a = 3249
    b = a // x
    c = b * x

    return a - c

def rabin_alt58(x) -> int:
    a = 3364
    b = a // x
    c = b * x

    return a - c

def rabin_alt59(x) -> int:
    a = 3481
    b = a // x
    c = b * x

    return a - c

def rabin_alt60(x) -> int:
    a = 3600
    b = a // x
    c = b * x

    return a - c

def rabin_alt61(x) -> int:
    a = 3721
    b = a // x
    c = b * x

    return a - c

def rabin_alt62(x) -> int:
    a = 3844
    b = a // x
    c = b * x

    return a - c

def rabin_alt63(x) -> int:
    a = 3969
    b = a // x
    c = b * x

    return a - c

def rabin_alt64(x) -> int:
    a = 4096
    b = a // x
    c = b * x

    return a - c
# %%
