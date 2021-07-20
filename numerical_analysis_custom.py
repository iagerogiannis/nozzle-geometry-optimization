import numpy as np
from math import log


def trapezoid(f_discrete, dx):
    return 0.5 * dx * sum([(f_discrete[i] + f_discrete[i + 1]) for i in range(len(f_discrete) - 1)])


def romberg(f_discrete, dx):
    def find_a_n():
        nonlocal a, n
        a = len(f_discrete) - 1
        while a % 2 == 0:
            a = a // 2
        n = int(log((len(f_discrete) - 1) // a, 2))

    a, n = 0, 0
    find_a_n()

    i_rom = [[0] * (n - i + 1) for i in range(n + 1)]

    for i in range(n + 1):
        i_rom[i][0] = trapezoid([f_discrete[2 ** (n - i) * j] for j in range(2 ** i * a + 1)], 2 ** (n - i) * dx)

    for j in range(1, n + 1):
        for i in range(n + 1 - j):
            i_rom[i][j] = (4 ** j * i_rom[i + 1][j - 1] - i_rom[i][j - 1]) / (4 ** j - 1)

    return i_rom[0][-1]


# Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def tdma_solver(a, b, c, d):
    """
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    """
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


def newton_raphson(f, df_dx, x0, error=1e-15):
    i = -1
    while abs(f(x0)) > error:
        i += 1
        x0 -= f(x0) / df_dx(x0)
    return x0


def secant(f, x0, x1, error=1e-15):
    fx0 = f(x0)
    fx1 = f(x1)
    i = -1
    while abs(fx1) > error:
        i += 1
        x2 = (x0 * fx1 - x1 * fx0) / (fx1 - fx0)
        x0, x1 = x1, x2
        fx0, fx1 = fx1, f(x2)
    return x1
