import sys

import numpy as np
from math import pi
from matplotlib import pyplot as plt

from numerical_analysis.splines.bezier import Bezier
from numerical_analysis.root_finding import secant, newton_raphson

from nozzle_geometry_optimization import numerical_analysis_custom as na
from export_lib import file_manager as fm
from export_lib.plot_lib import PlotExporter

np.set_printoptions(threshold=sys.maxsize)


class NozzleOptimizer:

    def __init__(self, initial_generatrix: Bezier, number_of_nodes, target_pressure, v0, free_directions=2,
                 datatype=np.float):

        # generatrix: Bezier Object
        self.plot_exporter = PlotExporter(self, "results/plots")
        self.global_iterations = -1
        self.__generatrix = initial_generatrix
        self.__free_directions = free_directions
        # n: Number of Design Variables
        self.__n = free_directions * (self.generatrix.n - 1)

        # v0: Velocity at the input
        self.__v0 = v0

        # target_pressure: p(x)
        self.__target_pressure = target_pressure

        self.__datatype = datatype

        # -------------------- Properties' Discretization --------------------

        self.__x_discretization = {"x": np.empty(number_of_nodes, dtype=self.datatype)}

        self.__generatrix_discretization = {"t": np.empty(number_of_nodes, dtype=self.datatype),
                                            "S": np.empty(number_of_nodes, dtype=self.datatype)}

        self.__target_pressure_discretization = {"p_tar": np.empty(number_of_nodes, dtype=self.datatype)}

        self.__primal_properties = {"v": np.empty(number_of_nodes, dtype=self.datatype),
                                    "p": np.empty(number_of_nodes, dtype=self.datatype)}

        self.__adjoint_properties = {"q": np.empty(number_of_nodes, dtype=self.datatype),
                                     "u": np.empty(number_of_nodes, dtype=self.datatype)}

        self.__first_order_derivatives = {"dS/db": np.empty([self.n, number_of_nodes], dtype=self.datatype),
                                          "dv/db": np.empty([self.n, number_of_nodes], dtype=self.datatype),
                                          "dp/db": np.empty([self.n, number_of_nodes], dtype=self.datatype)}

        self.__second_order_derivatives = {"d2S/db2": np.empty([self.n, self.n, number_of_nodes], dtype=self.datatype)}

        self.__f_objective = {"F": None}

        self.__sensitivity_derivatives = {"dF/db": np.empty(self.n, dtype=self.datatype),
                                          "d2F/db2": np.empty([self.n, self.n], dtype=self.datatype)}

        self.nodes = number_of_nodes
        self.__dx = 1 / (self.nodes - 1)

    # -- Discretizations -----------------------------------------------------------------------------------------------
    def discretize_x_direction(self):

        self.x_discretization = {"x": np.array([i * self.dx for i in range(self.nodes)], dtype=self.datatype)}

    def discretize_generatrix(self, method="newton-raphson", e=1e-12):

        methods = ["newton-raphson", "secant"]
        if method not in methods:
            raise ValueError("Invalid Method for calculation of inverse t(x). "
                             "Expected one of: {}".format(", ".join(methods)))

        if method == "newton-raphson":
            self.generatrix_discretization["t"] = np.array([newton_raphson(lambda ti: self.generatrix.x_t(ti) - xi,
                                                                              lambda ti: self.generatrix.dx_dt(ti), 0.5,
                                                                              e)
                                                            for xi in self.x_discretization["x"]], dtype=self.datatype)

        elif method == "secant":
            self.generatrix_discretization["t"] = np.array([secant(lambda ti: self.generatrix.x_t(ti) - xi, 0, 1, e)
                                                            for xi in self.x_discretization["x"]], dtype=self.datatype)

        if True in np.isnan(self.generatrix_discretization["t"]):
            return False

        self.generatrix_discretization["S"] = np.array([pi * self.generatrix.y_t(ti) ** 2
                                                        for ti in self.generatrix_discretization["t"]])

        return True

    def discretize_target_pressure(self):

        self.target_pressure_discretization = {"p_tar": np.array([self.target_pressure(xi) for xi in self.x_discretization["x"]],
                                                       dtype=self.datatype)}

    # -- Calculations --------------------------------------------------------------------------------------------------
    def calculate_f_objective(self):
        self.f_objective["F"] = na.romberg((self.primal_properties["p"] - self.target_pressure_discretization["p_tar"]) ** 2,
                                           self.dx) / 2
        return self.f_objective["F"]

    def solve_primal(self):

        self.primal_properties["v"] = self.v0 * self.generatrix_discretization["S"][0] / \
                                      self.generatrix_discretization["S"]

        self.primal_properties["p"] = 0.5 * (self.primal_properties["v"][-1] ** 2 - self.primal_properties["v"] ** 2)

    def solve_adjoint(self):

        def calculate_u():

            self.adjoint_properties["u"][0] = 0.

            du_dx_c = self.primal_properties["p"][0] - self.target_pressure_discretization["p_tar"][0]

            for i in range(self.nodes - 1):
                du_dx_n = self.primal_properties["p"][i + 1] - self.target_pressure_discretization["p_tar"][i + 1]
                self.adjoint_properties["u"][i + 1] = self.adjoint_properties["u"][i] + \
                                                      0.5 * self.dx * (du_dx_n + du_dx_c)
                du_dx_c = du_dx_n

        def calculate_q():

            self.adjoint_properties["q"][-1] = - self.primal_properties["v"][-1] * self.adjoint_properties["u"][-1] / \
                                               self.generatrix_discretization["S"][-1]

            dq_dx_c = - self.primal_properties["v"][-1] * (self.primal_properties["p"][-1] -
                                                           self.target_pressure_discretization["p_tar"][-1]) / \
                      self.generatrix_discretization["S"][-1]

            for i in np.arange(self.nodes - 2, -1, -1):
                dq_dx_p = - self.primal_properties["v"][i] * (self.primal_properties["p"][i] -
                                                              self.target_pressure_discretization["p_tar"][i]) / \
                          self.generatrix_discretization["S"][i]

                self.adjoint_properties["q"][i] = self.adjoint_properties["q"][i + 1] - \
                                                  0.5 * self.dx * (dq_dx_c + dq_dx_p)
                dq_dx_c = dq_dx_p

        calculate_u()
        calculate_q()

    def calculate_first_order_derivatives_of_primal(self, method="dd", e=1e-8):

        methods = ["dd", "complex step", "finite differences"]
        if method not in methods:
            raise ValueError("Invalid Method for calculation of Direct Differentiation. Expected one of: {}".
                             format(", ".join(methods)))

        if method == "dd":

            self.first_order_derivatives["dv/db"] = np.array([- self.primal_properties["v"] *
                                                              self.first_order_derivatives["dS/db"][n] /
                                                              self.generatrix_discretization["S"]
                                                              for n in range(self.n)],
                                                             dtype=self.datatype)

            self.first_order_derivatives["dp/db"] = np.array([- self.primal_properties["v"] *
                                                              self.first_order_derivatives["dv/db"][n]
                                                              for n in range(self.n)], dtype=self.datatype)

        elif method == "complex step":

            self.first_order_derivatives["dv/db"] = self.first_order_derivatives_complex_method("velocity", e)
            self.first_order_derivatives["dp/db"] = self.first_order_derivatives_complex_method("pressure", e)

        elif method == "finite differences":

            self.first_order_derivatives["dv/db"] = self.first_order_derivatives_finite_differences("velocity", e)
            self.first_order_derivatives["dp/db"] = self.first_order_derivatives_finite_differences("pressure", e)

        return {"dv/db": self.first_order_derivatives["dv/db"], "dp/db": self.first_order_derivatives["dp/db"]}

    def calculate_ds_db(self, method="analytical", e=1e-8):

        def ds_db_n_i_analytical(n, i):

            a, b = self.n_to_cp_indices(n)
            t = self.generatrix_discretization["t"][i]

            if b == 0:
                value = - ((2 * pi * self.generatrix.y_t(t)) * self.generatrix.dy_dt(t) *
                           self.generatrix.dx_dt(t) ** (-1) * self.generatrix.p_i(n + 1, t))
            else:
                value = ((2 * pi * self.generatrix.y_t(t)) * self.generatrix.p_i(a, t))

            return value

        methods = ["analytical", "complex step", "finite differences"]
        if method not in methods:
            raise ValueError("Invalid Method. Expected one of: {}".format(", ".join(methods)))

        if method == "analytical":
            self.first_order_derivatives["dS/db"] = np.array([[ds_db_n_i_analytical(n, i) for i in range(self.nodes)]
                                                              for n in range(self.n)], dtype=self.datatype)

        elif method == "complex step":
            self.first_order_derivatives["dS/db"] = self.first_order_derivatives_complex_method("cross-section area", e)

        elif method == "finite differences":
            self.first_order_derivatives["dS/db"] = \
                self.first_order_derivatives_finite_differences("cross-section area", e)

        return self.first_order_derivatives["dS/db"]

    def calculate_d2s_db2(self, method="analytical",  ex=1e-4, ey=1e-1):

        def d2s_dbn_dbm_i_analytical(n, m, i):

            t = self.generatrix_discretization["t"][i]

            a, b = self.n_to_cp_indices(n)
            c, d = self.n_to_cp_indices(m)

            if b == 0 and d == 0:

                return 2 * pi * self.generatrix.y_t(t) * self.generatrix.dx_dt(t) ** (-2) * self.generatrix.dy_dt(t) * \
                       self.generatrix.p_i(a, t) * self.generatrix.polynomials["p"][c].derivative().value(t)

            elif b == 1 and d == 1:

                return 2 * pi * self.generatrix.p_i(a, t) * self.generatrix.p_i(c, t)

            else:

                if b == 0:

                    return - 2 * pi * self.generatrix.dx_dt(t) ** (-1) * self.generatrix.p_i(a, t) * \
                           (self.generatrix.dy_dt(t) * self.generatrix.p_i(c, t) + self.generatrix.y_t(t) *
                            self.generatrix.polynomials["p"][c].derivative().value(t))

                else:

                    return - 2 * pi * self.generatrix.dx_dt(t) ** (-1) * self.generatrix.p_i(c, t) * \
                           (self.generatrix.dy_dt(t) * self.generatrix.p_i(a, t) + self.generatrix.y_t(t) *
                            self.generatrix.polynomials["p"][a].derivative().value(t))

        methods = ["analytical", "finite differences"]
        if method not in methods:
            raise ValueError("Invalid Method for calculation of second derivatives of cross-section area. "
                             "Expected one of: {}".format(", ".join(methods)))

        if method == "analytical":

            for m in range(self.n):
                for n in range(self.n):
                    if m < n + 1:
                        for i in range(self.nodes):
                            self.second_order_derivatives["d2S/db2"][n, m, i] = d2s_dbn_dbm_i_analytical(n, m, i)
                    else:
                        self.second_order_derivatives["d2S/db2"][n, m] = self.second_order_derivatives["d2S/db2"][m, n]

        elif method == "finite differences":
            self.second_order_derivatives["d2S/db2"] = \
                self.second_order_derivatives_finite_differences("cross-section area", ex, ey)

        return self.second_order_derivatives["d2S/db2"]

    def calculate_grad(self, method="adjoint", e=1e-8):

        def direct_method():

            self.sensitivity_derivatives["dF/db"] = np.array([na.romberg((self.primal_properties["p"] -
                                                                          self.target_pressure_discretization["p_tar"]) *
                                                                         self.first_order_derivatives["dp/db"][n],
                                                                         self.dx) for n in range(self.n)],
                                                             dtype=self.datatype)

        def adjoint_method():

            self.sensitivity_derivatives["dF/db"] = np.array([na.romberg((self.primal_properties["v"] ** 2 /
                                                                          self.generatrix_discretization["S"]) *
                                                                         (self.primal_properties["p"] -
                                                                          self.target_pressure_discretization["p_tar"]) *
                                                                         self.first_order_derivatives["dS/db"][n],
                                                                         self.dx) for n in range(self.n)],
                                                             dtype=self.datatype)

        def complex_variables_method():
            self.sensitivity_derivatives["dF/db"] = self.first_order_derivatives_complex_method("f_objective", e)

        def finite_differences_method():
            self.sensitivity_derivatives["dF/db"] = self.first_order_derivatives_finite_differences("f_objective", e)

        methods = ["direct", "adjoint", "complex step", "finite differences"]
        if method not in methods:
            raise ValueError("Invalid Method for calculation of Jacobian Matrix. Expected one of: {}".
                             format(", ".join(methods)))

        if method == "direct":
            direct_method()
        elif method == "adjoint":
            adjoint_method()
        elif method == "complex step":
            complex_variables_method()
        elif method == "finite differences":
            finite_differences_method()

        return self.sensitivity_derivatives["dF/db"]

    def calculate_hessian(self, method="dd-av", ex=1e-6, ey=1e-6):

        def term(n, m):
            return (self.adjoint_properties["u"] * d_dv_db_dx[m] + self.primal_properties["v"] *
                    (self.primal_properties["p"] - self.target_pressure_discretization["p_tar"]) *
                    self.first_order_derivatives["dS/db"][m] / self.generatrix_discretization["S"]) * \
                   self.first_order_derivatives["dv/db"][n]

        methods = ["dd-av", "finite differences"]
        if method not in methods:
            raise ValueError("Invalid Method for calculation of Hessian Matrix. Expected one of: {}".
                             format(", ".join(methods)))

        if method == "dd-av":

            d_dv_db_dx = np.array([self.apply_d_dx(self.first_order_derivatives["dv/db"][n]) for n in range(self.n)],
                                  dtype=self.datatype)

            for m in range(self.n):
                for n in range(self.n):
                    if m > n:
                        self.sensitivity_derivatives["d2F/db2"][n, m] = self.sensitivity_derivatives["d2F/db2"][m, n]
                    else:
                        self.sensitivity_derivatives["d2F/db2"][n, m] = \
                            na.romberg((self.primal_properties["v"] ** 2 * (self.primal_properties["p"] -
                                        self.target_pressure_discretization["p_tar"]) / self.generatrix_discretization["S"]) *
                                        self.second_order_derivatives["d2S/db2"][n, m] +
                                       self.first_order_derivatives["dp/db"][n] *
                                       self.first_order_derivatives["dp/db"][m] + term(n, m) + term(m, n), self.dx)

        elif method == "finite differences":
            self.sensitivity_derivatives["d2F/db2"] = \
                self.second_order_derivatives_finite_differences("f_objective", ex, ey)

        return self.sensitivity_derivatives["d2F/db2"]

    def first_order_derivatives_finite_differences(self, property_name, e=1e-8):

        def execute_require_actions():
            nonlocal required_actions
            for action in required_actions:
                action()

        var_name = ""
        key = ""
        result = None

        properties = ["velocity", "pressure", "cross-section area", "f_objective"]
        if property_name not in properties:
            raise ValueError("Invalid Property Name. Expected one of: {}".format(", ".join(properties)))

        if property_name == "velocity":
            var_name = "primal_properties"
            key = "v"
            required_actions = [self.solve_primal]
            result = np.empty([self.n, self.nodes], dtype=self.datatype)
        elif property_name == "pressure":
            var_name = "primal_properties"
            key = "p"
            required_actions = [self.solve_primal]
            result = np.empty([self.n, self.nodes], dtype=self.datatype)
        elif property_name == "cross-section area":
            var_name = "generatrix_discretization"
            key = "S"
            required_actions = []
            result = np.empty([self.n, self.nodes], dtype=self.datatype)
        elif property_name == "f_objective":
            var_name = "f_objective"
            key = "F"
            required_actions = [self.solve_primal, self.calculate_f_objective]
            result = np.empty(self.n, dtype=self.datatype)

        for n in range(self.n):
            i, k = self.n_to_cp_indices(n)

            cpn_init = self.generatrix.cp[i, k]

            self.modify_generatrix_cp_coordinate(i, k, e)
            execute_require_actions()
            f_plus = getattr(self, var_name)[key]

            self.modify_generatrix_cp_coordinate(i, k, cpn_init, "absolute")
            self.modify_generatrix_cp_coordinate(i, k, -e)
            execute_require_actions()
            f_minus = getattr(self, var_name)[key]

            self.modify_generatrix_cp_coordinate(i, k, cpn_init, "absolute")
            result[n] = 0.5 * (f_plus - f_minus) / e

        return result

    def first_order_derivatives_complex_method(self, property_name, e=1e-30):

        def execute_require_actions():
            nonlocal required_actions
            for action in required_actions:
                action()

        var_name = ""
        key = ""
        result = None

        properties = ["velocity", "pressure", "cross-section area", "f_objective"]
        if property_name not in properties:
            raise ValueError("Invalid Property Name. Expected one of: {}".format(", ".join(properties)))

        if property_name == "velocity":
            var_name = "primal_properties"
            key = "v"
            required_actions = [self.solve_primal]
            result = np.empty([self.n, self.nodes], dtype=self.datatype)
        elif property_name == "pressure":
            var_name = "primal_properties"
            key = "p"
            required_actions = [self.solve_primal]
            result = np.empty([self.n, self.nodes], dtype=self.datatype)
        elif property_name == "cross-section area":
            var_name = "generatrix_discretization"
            key = "S"
            required_actions = []
            result = np.empty([self.n, self.nodes], dtype=self.datatype)
        elif property_name == "f_objective":
            var_name = "f_objective"
            key = "F"
            required_actions = [self.solve_primal, self.calculate_f_objective]
            result = np.empty(self.n, dtype=self.datatype)

        for n in range(self.n):
            i, k = self.n_to_cp_indices(n)

            self.modify_generatrix_cp_coordinate(i, k, e * complex(0, 1))
            execute_require_actions()
            result[n] = getattr(self, var_name)[key].imag / e

            self.modify_generatrix_cp_coordinate(i, k, - e * complex(0, 1))
            self.generatrix.modify_control_point_coordinate(i, k, self.generatrix.cp[i][k].real)

        return result

    def second_order_derivatives_finite_differences(self, property_name, ex=1e-1, ey=1e-1):

        def execute_require_actions():
            nonlocal required_actions
            for action in required_actions:
                action()

        def fij(i, j):
            nonlocal a, b, c, d, en, em
            cpn_init = self.generatrix.cp[a, b]
            cpm_init = self.generatrix.cp[c, d]
            self.modify_generatrix_cp_coordinate(a, b, i * en)
            self.modify_generatrix_cp_coordinate(c, d, j * em)
            execute_require_actions()
            value = getattr(self, var_name)[key]
            self.modify_generatrix_cp_coordinate(a, b, cpn_init, "absolute")
            self.modify_generatrix_cp_coordinate(c, d, cpm_init, "absolute")
            return value

        var_name = ""
        key = ""
        result = None

        properties = ["cross-section area", "f_objective"]
        if property_name not in properties:
            raise ValueError("Invalid Property Name. Expected one of: {}".format(", ".join(properties)))

        if property_name == "cross-section area":
            var_name = "generatrix_discretization"
            key = "S"
            result = np.empty([self.n, self.n, self.nodes], dtype=self.datatype)
            required_actions = []
        elif property_name == "f_objective":
            var_name = "f_objective"
            key = "F"
            required_actions = [self.solve_primal, self.calculate_f_objective]
            result = np.empty([self.n, self.n], dtype=self.datatype)

        for n in range(self.n):
            for m in range(self.n):
                if n > m:
                    result[n, m] = result[m, n]
                else:

                    a, b = self.n_to_cp_indices(n)
                    c, d = self.n_to_cp_indices(m)

                    if b == 0:
                        en = ex
                    else:
                        en = ey
                    if d == 0:
                        em = ex
                    else:
                        em = ey

                    f1 = fij(1, 1)
                    f2 = fij(1, -1)
                    f3 = fij(-1, 1)
                    f4 = fij(-1, -1)

                    result[n, m] = (f1 - f2 - f3 + f4) / (4 * en * em)

        return result

    # -- Optimization Methods ------------------------------------------------------------------------------------------
    def steepest_descent(self, iterations, step=1.25e-8, method_grad="adjoint", method_ds_db="analytical",
                         method_dd="dd", e_grad=1e-5, e_ds_db=1e-5, e_dd=1e-5, log_fname="log.dat"):

        fm.create_log_file(log_fname, "Steepest Decent", self.free_directions, self.nodes, step, method_dd,
                           method_ds_db, "*", method_grad, "*", e_dd, e_ds_db, "*", "*", e_grad, "*", "*")

        fm.write_log_file_headers(log_fname, self.generatrix.n + 1, self.free_directions)

        self.solve_primal()
        self.calculate_f_objective()

        tu = 1
        tu_t = 1

        print("{:4} | {}".format(0, self.f_objective["F"].real))
        fm.write_optimization_cycle_data(log_fname, 0, tu, tu_t, self.f_objective["F"].real, self.generatrix.cp)

        for i in range(iterations):
            self.global_iterations += 1
            tu = 0

            self.plot_exporter.export_plot("{}_gen".format(str(self.global_iterations).zfill(4)),
                                           "Iteration {} (Method: {})".format(str(self.global_iterations), "Steepest Descent"))

            if method_grad == "adjoint":
                self.calculate_ds_db(method_ds_db, e_ds_db)
                tu += 1

            if method_grad == "direct":
                self.calculate_ds_db(method_ds_db, e_ds_db)
                self.calculate_first_order_derivatives_of_primal(method_dd, e_dd)
                tu += 1

            self.calculate_grad(method_grad, e_grad)
            self.apply_db_on_cp(- step * self.sensitivity_derivatives["dF/db"])

            self.solve_primal()
            self.calculate_f_objective()
            tu += 1

            tu_t += tu
            print("{:4} | {}".format(i + 1, self.f_objective["F"].real))
            fm.write_optimization_cycle_data(log_fname, i + 1, tu, tu_t, self.f_objective["F"].real, self.generatrix.cp)

    def newton_method(self, iterations, method_grad="adjoint", method_hessian="dd-av", method_dd="dd",
                      method_ds_db="analytical", method_d2s_db2="analytical",
                      e_grad=1e-8, ex_hessian=1e-6, ey_hessian=1e-6, e_dd=1e-30, e_ds_db=1e-5,
                      ex_d2s_db2=1e-4, ey_d2s_db2=1e-1, log_fname="log.dat"):

        fm.create_log_file(log_fname, "Newton", self.free_directions, self.nodes, "*", method_dd, method_ds_db,
                           method_d2s_db2, method_grad, method_hessian, e_dd, e_ds_db, ex_d2s_db2, ey_d2s_db2,
                           e_grad, ex_hessian, ey_hessian)

        fm.write_log_file_headers(log_fname, self.generatrix.n + 1, self.free_directions)

        self.solve_primal()
        self.calculate_f_objective()

        tu = 1
        tu_t = 1

        print("{:4} | {}".format(0, self.f_objective["F"].real))
        fm.write_optimization_cycle_data(log_fname, 0, tu, tu_t, self.f_objective["F"].real, self.generatrix.cp)

        for i in range(iterations):

            self.global_iterations += 1
            tu = 0

            if method_grad == "adjoint" or method_grad == "direct" or method_hessian == "dd-av":
                self.calculate_ds_db(method_ds_db, e_ds_db)

            if method_hessian == "dd-av" or method_grad == "direct":
                self.calculate_first_order_derivatives_of_primal(method_dd, e_dd)
                tu += self.n

            if method_hessian == "dd-av":
                self.solve_adjoint()
                self.calculate_d2s_db2(method_d2s_db2, ex_d2s_db2, ey_d2s_db2)
                tu += 1

            self.calculate_grad(method_grad, e_grad)
            self.calculate_hessian(method_hessian, ex_hessian, ey_hessian)

            db = - np.matmul(np.linalg.inv(self.sensitivity_derivatives["d2F/db2"]),
                             self.sensitivity_derivatives["dF/db"])

            while not self.generatrix_is_one_to_one(db):
                db *= 0.75

            self.apply_db_on_cp(db)

            self.solve_primal()
            self.calculate_f_objective()
            tu += 1

            tu_t += tu
            self.plot_exporter.export_plot("{}_gen".format(str(self.global_iterations).zfill(4)),
                                           "Iteration {} (Method: {})".format(str(self.global_iterations), "Newton's"))
            print("{:4} | {}".format(i + 1, self.f_objective["F"].real))
            fm.write_optimization_cycle_data(log_fname, i + 1, tu, tu_t, self.f_objective["F"].real, self.generatrix.cp)

    def bfgs_method(self, iterations, method_grad="adjoint", method_ds_db="analytical",
                         method_dd="dd", e_grad=1e-5, e_ds_db=1e-5, e_dd=1e-5, log_fname="log.dat"):

        def steepest_decent_step():

            cp_init = self.generatrix.cp.copy()
            step = 2.5e-8

            self.solve_primal()
            self.calculate_ds_db()
            self.calculate_grad()
            while not self.apply_db_on_cp(- step * self.sensitivity_derivatives["dF/db"]):
                self.apply_db_on_cp(cp_init, "absolute")
                step *= .5
            self.solve_primal()
            self.calculate_f_objective()

            return - step * self.sensitivity_derivatives["dF/db"]

        def line_search(a=1., c=0.5, tau=0.5):

            nonlocal pk

            a /= tau
            j = -1

            f0 = self.f_objective["F"]

            while True:

                j += 1
                if j > 40:
                    break

                a *= tau

                cp_init = self.generatrix.cp.copy()

                if not self.generatrix_is_one_to_one(a * pk):
                    continue

                if not self.apply_db_on_cp(a * pk):
                    self.apply_db_on_cp(cp_init, "absolute")
                    continue

                self.solve_primal()
                self.calculate_f_objective()
                f1 = self.f_objective["F"]

                if f1 - f0 - c * a * self.sensitivity_derivatives["dF/db"].dot(pk) <= 0:
                    # if f1 / f0 > 0.99:
                    #     self.apply_db_on_cp(cp_init, "absolute")
                    #     return steepest_decent_step()
                    return a * pk

                self.apply_db_on_cp(cp_init, "absolute")

            return None

        fm.create_log_file(log_fname, "BFGS", self.free_directions, self.nodes, "*", method_dd, method_ds_db,
                           "*", method_grad, "*", e_dd, e_ds_db, "*", "*", e_grad, "*", "*")

        fm.write_log_file_headers(log_fname, self.generatrix.n + 1, self.free_directions)

        self.solve_primal()
        self.calculate_f_objective()

        tu = 1
        tu_t = 1

        print("{:4} | {}".format(0, self.f_objective["F"].real))
        fm.write_optimization_cycle_data(log_fname, 0, tu, tu_t, self.f_objective["F"].real, self.generatrix.cp)

        hessian_approximation_inverse = np.identity(self.n)

        for i in range(iterations):
            self.global_iterations += 1
            self.calculate_ds_db()
            self.calculate_grad()
            y0 = self.sensitivity_derivatives["dF/db"]

            pk = - np.matmul(hessian_approximation_inverse, self.sensitivity_derivatives["dF/db"])

            sk = line_search()
            if sk is None:
                print("Steepest Descent Step")
                self.steepest_descent(10)
            else:
                self.calculate_ds_db()
                self.calculate_grad()
                y1 = self.sensitivity_derivatives["dF/db"]

                yk = y1 - y0

                hessian_approximation_inverse = hessian_approximation_inverse + \
                                                ((sk.dot(yk) + yk.dot(np.matmul(hessian_approximation_inverse, yk))) /
                                                 (sk.dot(yk)) ** 2) * \
                                                np.matmul(sk.reshape((self.n, 1)), sk.reshape((1, self.n))) - \
                                                (1 / sk.dot(yk)) * \
                                                (np.matmul(hessian_approximation_inverse,
                                                           np.matmul(yk.reshape((self.n, 1)), sk.reshape((1, self.n)))) +
                                                 np.matmul(sk.reshape((self.n, 1)), np.matmul(yk.reshape((1, self.n)),
                                                                                            hessian_approximation_inverse)))

                self.solve_primal()
                self.calculate_f_objective()
                tu += 1

                tu_t += tu

                self.plot_exporter.export_plot("{}_gen".format(str(self.global_iterations).zfill(4)),
                                               "Iteration {} (Method: {})".format(str(self.global_iterations), "BFGS"))

                print("{:4} | {}".format(i + 1, self.f_objective["F"].real))
                fm.write_optimization_cycle_data(log_fname, i + 1, tu, tu_t, self.f_objective["F"].real, self.generatrix.cp)

    # -- Bezier Interaction Functions ----------------------------------------------------------------------------------
    def modify_generatrix_cp_coordinate(self, i, k, dl, method="relative"):

        methods = ["relative", "absolute"]
        if method not in methods:
            raise ValueError("Invalid Method Name. Expected one of: {}".format(", ".join(methods)))

        if method == "relative":
            self.generatrix.modify_control_point_coordinate(i, k, self.generatrix.cp[i, k] + dl)
        elif method == "absolute":
            self.generatrix.modify_control_point_coordinate(i, k, dl)
        self.discretize_generatrix()

    def generatrix_is_one_to_one(self, db):

        dt = 1e-3

        for n in range(self.n):
            j, k = self.n_to_cp_indices(n)
            self.generatrix.modify_control_point_coordinate(j, k, self.generatrix.cp[j, k] + db[n])

        xc = self.generatrix.x_t(0)

        is_one_to_one = True

        for t in np.arange(dt, 1. + dt, dt):
            xn = self.generatrix.x_t(t)
            if xn - xc <= 1e-14:
                is_one_to_one = False
                break
            else:
                xc = xn

        for n in range(self.n):
            j, k = self.n_to_cp_indices(n)
            self.generatrix.modify_control_point_coordinate(j, k, self.generatrix.cp[j, k] - db[n])

        return is_one_to_one

    def apply_db_on_cp(self, db, method="relative"):

        methods = ["relative", "absolute"]
        if method not in methods:
            raise ValueError("Invalid Method Name. Expected one of: {}".format(", ".join(methods)))

        if method == "relative":
            for n in range(self.n):
                j, k = self.n_to_cp_indices(n)
                self.generatrix.modify_control_point_coordinate(j, k, self.generatrix.cp[j, k] + db[n])
        elif method == "absolute":
            for n in range(self.n):
                j, k = self.n_to_cp_indices(n)
                self.generatrix.modify_control_point_coordinate(j, k, db[j, k])

        return self.discretize_generatrix()

    # -- Secondary Functions -------------------------------------------------------------------------------------------
    def apply_d_dx(self, array):

        d_array_dx = np.empty(self.nodes, dtype=self.datatype)

        d_array_dx[0] = (array[1] - array[0]) / self.dx
        for i in range(1, self.nodes - 1):
            d_array_dx[i] = (array[i + 1] - array[i - 1]) / (2 * self.dx)
        d_array_dx[-1] = (array[-1] - array[-2]) / self.dx

        return d_array_dx

    def n_to_cp_indices(self, n):

        if self.free_directions == 2:
            if n < self.n / 2:
                return n + 1, 0
            else:
                return n + 1 - self.n // 2, 1
        else:
            return n + 1, 1

    def change_free_directions(self, free_directions):

        self.free_directions = free_directions
        self.n = free_directions * (self.generatrix.n - 1)

        self.first_order_derivatives = {"dS/db": np.empty([self.n, self.nodes], dtype=self.datatype),
                                        "dv/db": np.empty([self.n, self.nodes], dtype=self.datatype),
                                        "dp/db": np.empty([self.n, self.nodes], dtype=self.datatype)}

        self.second_order_derivatives = {"d2S/db2": np.empty([self.n, self.n, self.nodes], dtype=self.datatype)}

        self.sensitivity_derivatives = {"dF/db": np.empty(self.n, dtype=self.datatype),
                                        "d2F/db2": np.empty([self.n, self.n], dtype=self.datatype)}

    def change_datatype(self, datatype):

        self.datatype = datatype

        for dictionary in [self.x_discretization, self.generatrix_discretization, self.target_pressure_discretization,
                           self.primal_properties, self.adjoint_properties, self.first_order_derivatives,
                           self.second_order_derivatives, self.sensitivity_derivatives]:
            for key in dictionary.keys():
                dictionary[key] = dictionary[key].astype(datatype)

        self.generatrix.change_datatype(datatype)

    # -- Results Presentations Functions -------------------------------------------------------------------------------
    def draw_pressures(self):
        plt.plot(self.x_discretization["x"], self.primal_properties["p"],
                 self.x_discretization["x"], self.target_pressure_discretization["p_tar"])
        plt.show()

    def draw_duct(self):
        graph = self.generatrix.graph(0.001)
        graph_symmetric = [graph[0], [-value for value in graph[1]]]
        plt.plot(graph[0], graph[1], graph_symmetric[0], graph_symmetric[1])
        plt.gca().set_aspect('equal')
        plt.show()

    # -- Data Files Export Functions -----------------------------------------------------------------------------------
    def export_generatrix(self, filename="generatrix.dat"):
        graph = self.generatrix.graph(0.001)
        f = open(filename, "w")
        f.write("# {:25}{:25}\n".format("x [m]", "y [m]"))
        for i in range(len(graph[0])):
            f.write("  {:25}{:25}\n".format(str(graph[0][i]), str(graph[1][i])))
        f.close()

    def export_pressure(self, filename="pressure.dat"):
        f = open(filename, "w")
        f.write("# {:25}{:25}\n".format("x [m]", "pressure [Pa]"))
        for i in range(self.nodes):
            f.write("  {:25}{:25}\n".format(str(self.x_discretization["x"][i]), str(self.primal_properties["p"][i])))
        f.close()

    def export_target_pressure(self, filename="target_pressure.dat"):
        f = open(filename, "w")
        f.write("# {:25}{:25}\n".format("x [m]", "target_pressure [Pa]"))
        for i in range(self.nodes):
            f.write("  {:25}{:25}\n".format(str(self.x_discretization["x"][i]),
                                            str(self.target_pressure_discretization["p_tar"][i])))
        f.close()

    def export_velocity(self, filename="velocity.dat"):
        f = open(filename, "w")
        f.write("# {:25}{:25}\n".format("x [m]", "velocity [Pa]"))
        for i in range(self.nodes):
            f.write("  {:25}{:25}\n".format(str(self.x_discretization["x"][i]), str(self.primal_properties["v"][i])))
        f.close()

    # -- Class Properties ----------------------------------------------------------------------------------------------
    @property
    def generatrix(self):
        return self.__generatrix

    @property
    def n(self):
        return self.__n

    @property
    def v0(self):
        return self.__v0

    @property
    def target_pressure(self):
        return self.__target_pressure

    @property
    def nodes(self):
        return self.__nodes

    @property
    def dx(self):
        return self.__dx

    @property
    def x_discretization(self):
        return self.__x_discretization

    @property
    def generatrix_discretization(self):
        return self.__generatrix_discretization

    @property
    def target_pressure_discretization(self):
        return self.__target_pressure_discretization

    @property
    def f_objective(self):
        return self.__f_objective

    @property
    def primal_properties(self):
        return self.__primal_properties

    @property
    def adjoint_properties(self):
        return self.__adjoint_properties

    @property
    def first_order_derivatives(self):
        return self.__first_order_derivatives

    @property
    def second_order_derivatives(self):
        return self.__second_order_derivatives

    @property
    def sensitivity_derivatives(self):
        return self.__sensitivity_derivatives

    @property
    def free_directions(self):
        return self.__free_directions

    @property
    def datatype(self):
        return self.__datatype

    @n.setter
    def n(self, n):
        self.__n = n

    @nodes.setter
    def nodes(self, nodes):
        self.__nodes = nodes

        self.dx = 1 / (nodes - 1)

        self.discretize_x_direction()
        self.discretize_generatrix()
        self.discretize_target_pressure()

    @dx.setter
    def dx(self, dx):
        self.__dx = dx

    @x_discretization.setter
    def x_discretization(self, x_discretization):
        self.__x_discretization = x_discretization

    @generatrix_discretization.setter
    def generatrix_discretization(self, generatrix_discretization):
        self.__generatrix_discretization = generatrix_discretization

    @target_pressure_discretization.setter
    def target_pressure_discretization(self, target_pressure_discretization):
        self.__target_pressure_discretization = target_pressure_discretization

    @f_objective.setter
    def f_objective(self, f_objective):
        self.f_objective = f_objective

    @primal_properties.setter
    def primal_properties(self, primal_properties):
        self.__primal_properties = primal_properties

    @adjoint_properties.setter
    def adjoint_properties(self, adjoint_properties):
        self.__adjoint_properties = adjoint_properties

    @first_order_derivatives.setter
    def first_order_derivatives(self, first_order_derivatives):
        self.__first_order_derivatives = first_order_derivatives

    @second_order_derivatives.setter
    def second_order_derivatives(self, second_order_derivatives):
        self.__second_order_derivatives = second_order_derivatives

    @sensitivity_derivatives.setter
    def sensitivity_derivatives(self, sensitivity_derivatives):
        self.__sensitivity_derivatives = sensitivity_derivatives

    @free_directions.setter
    def free_directions(self, free_directions):
        self.__free_directions = free_directions

    @datatype.setter
    def datatype(self, datatype):
        self.__datatype = datatype
