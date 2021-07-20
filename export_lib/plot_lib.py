import os
import matplotlib.pyplot as plt
import numpy as np


class PlotExporter:

    def __init__(self, optimizer, path=None, file_type="png"):
        self.optimizer = optimizer
        self.fig, (self.ax1, self.ax2) = plt.subplots(2)
        self.file_type = file_type
        if not path:
            self.path = ""
        else:
            self.path = path
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def plot_nozzle(self):
        u = np.array(self.optimizer.generatrix.graph(0.01))
        d = np.array([np.array(self.optimizer.generatrix.graph(0.01))[0],
                      -np.array(self.optimizer.generatrix.graph(0.01))[1]])

        self.ax1.plot(*u, "k")
        self.ax1.plot(*d, "k")
        self.ax1.plot([0., 1.], [0., 0.], "-.k")

    def plot_pressure_distribution(self):
        self.ax2.plot(self.optimizer.x_discretization["x"], self.optimizer.primal_properties["p"])
        self.ax2.plot(self.optimizer.x_discretization["x"], self.optimizer.target_pressure_discretization["p_tar"])
        self.ax2.legend(["Pressure Distr", "Target Pressure Distr"], loc="lower right")

    def create_figure(self, title=""):
        self.fig.suptitle(title)
        self.plot_nozzle()
        self.plot_pressure_distribution()

    def show_plot(self, title=""):
        self.create_figure(title)
        plt.show()

    def export_plot(self, fname, title=""):
        self.create_figure(title)
        fname = r"{}/{}.{}".format(self.path, fname, self.file_type)
        plt.savefig(fname)
        self.ax1.clear()
        self.ax2.clear()
