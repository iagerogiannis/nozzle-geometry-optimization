def create_log_file(fname, optimization_method, directions, nodes, step, method_dd, method_ds_db, method_d2s_db2,
                    method_jacobian, method_hessian, e_dd, e_ds_db, ex_d2s_db2, ey_d2s_db2, e_jacobian,
                    ex_hessian, ey_hessian):

    format1 = "# {:71}: {}\n"
    format2 = "# {:71}: {:25}| e  : {:10}\n"
    format3 = "# {:71}: {:25}| e  : {:10}| ey : {:10}\n"

    # Clear file
    open(fname, 'w').close()

    f = open(fname, "a")

    f.write(format1.format("Method of Optimization", optimization_method))
    f.write(format1.format("Directions of Optimization", directions))
    f.write(format1.format("Number of Nodes", nodes))

    if optimization_method == "Steepest Decent":
        f.write(format1.format("Step", step))

    if (optimization_method == "Newton" and method_hessian == "dd-av") or method_jacobian == "direct":
        if method_dd == "finite differences":
            f.write(format2.format("Method of first derivatives of Velocity and Pressure Calculation", method_dd, str(e_dd)))
        else:
            f.write(format1.format("Method of first derivatives of Velocity and Pressure Calculation", method_dd))

    if method_jacobian != "finite differences" and method_hessian != "finite differences":
        if method_ds_db == "finite differences":
            f.write(format2.format("Method of first derivatives of Cross-Section Area Calculation", method_ds_db, str(e_ds_db)))
        else:
            f.write(format1.format("Method of first derivatives of Cross-Section Area Calculation", method_ds_db))

    if optimization_method == "Newton" and method_hessian == "dd-av":
        if method_d2s_db2 == "finite differences":
            f.write(format3.format("Method of second derivatives of Cross-Secion Area Calculation",
                                   method_d2s_db2, str(ex_d2s_db2), str(ey_d2s_db2)))
        else:
            f.write(format1.format("Method of second derivatives of Cross-Secion Area Calculation", method_d2s_db2))

    if method_jacobian == "finite differences":
        f.write(format2.format("Method of Jacobian Calculation", method_jacobian, str(e_jacobian)))
    else:
        f.write(format1.format("Method of Jacobian Calculation", method_jacobian))

    if optimization_method == "Newton":
        if method_hessian == "finite differences":
            f.write(format3.format("Method of Hessian Calculation", method_hessian, str(ex_hessian), str(ey_hessian)))
        else:
            f.write(format1.format("Method of Hessian Calculation", method_hessian))

    f.write("\n")
    f.close()


def write_log_file_headers(fname, num_of_cp, directions):

    x = ["X" + str(i) for i in range(num_of_cp)]
    y = ["Y" + str(i) for i in range(num_of_cp)]

    for i in range(num_of_cp - 2):
        if directions == 1:
            y[i + 1] += " (b" + str(i) + ")"
        else:
            x[i + 1] += " (b" + str(i) + ")"
            y[i + 1] += " (b" + str(i + num_of_cp - 2) + ")"

    f = open(fname, "a")

    f.write("# {:25}{:25}{:25}{:25}".format("Cycle of Optimization", "Time Units of Cycle", "Total Time Units",
                                            "F Objective") + str(num_of_cp * "{:25}").format(*x) +
                                                             str(num_of_cp * "{:25}").format(*y) + "\n\n")

    f.close()


def write_optimization_cycle_data(fname, cycle, time_units, total_time_units, f_objective, cp):

    f = open(fname, "a")

    f.write("  {:25}{:25}{:25}{:25}".format(str(cycle), str(time_units), str(total_time_units), str(f_objective)) +
            str(len(cp) * "{:25}").format(*[str(cp[i, 0].real) for i in range(len(cp))]) +
            str(len(cp) * "{:25}").format(*[str(cp[i, 1].real) for i in range(len(cp))]) + "\n")

    f.close()

