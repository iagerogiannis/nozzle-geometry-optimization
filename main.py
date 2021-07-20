import numpy as np

from numerical_analysis.splines.bezier import Bezier
from numerical_analysis.root_finding import secant

from nozzle_optimizer import NozzleOptimizer
from matplotlib import pyplot as plt

np.set_printoptions(linewidth=np.inf)


def p_target(x):
    p_target_bezier = Bezier(np.array([[0, 187.5], [0.15, 187.5], [0.2, -400], [0.15, -2500], [0.3, 0], [0.9, 0],
                                       [1, 0]]))
    return p_target_bezier.y_t(secant(lambda ti: p_target_bezier.x_t(ti) - x, 0, 1))


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


CP = np.array([[0., 0.4],
               [0.05, 0.35],
               [0.05, 0.15],
               [0.05, 0.1],
               [0.1, 0.1],
               [0.2, 0.15],
               [0.5, 0.15],
               [0.65, 0.2],
               [1., 0.2]])


curve = Bezier(CP.copy())
solver = NozzleOptimizer(curve, 101, p_target, 5.)

a = 30
b = 6

for i in range(2):
    solver.steepest_descent(a - b)
    solver.bfgs_method(b)
solver.steepest_descent(100, step=1.5e-8)

solver.steepest_descent(1)

# print(solver.generatrix.cp)
u = np.array(solver.generatrix.graph(0.01))
d = np.array([np.array(solver.generatrix.graph(0.01))[0], -np.array(solver.generatrix.graph(0.01))[1]])

plt.plot(*u, "b")
plt.plot(*d, "b")
plt.show()
