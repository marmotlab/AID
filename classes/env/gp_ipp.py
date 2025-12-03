import warnings
from itertools import product

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C

from classes.env.Gaussian2D import Gaussian2D


class GaussianProcessForIPP():
    def __init__(self, threshold=0.4, beta=1.0, sensor_range=0.45):
        # self.kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
        # self.kernel = RBF(0.2)
        self.kernel = Matern(length_scale=sensor_range)   # default 0.45
        self.gp = GaussianProcessRegressor(kernel=self.kernel, optimizer=None, n_restarts_optimizer=0)
        self.threshold = threshold
        self.beta = beta
        self.sensor_range = sensor_range
        self.observed_points = []
        self.observed_value = []

    def add_observed_point(self, point_pos, value):
        self.observed_points.append(point_pos)
        self.observed_value.append(value)
        # print(f"observation points are {len(self.observed_points)}")
        # print(f"observation values are {len(self.observed_value)}")

    def flexi_updates(self, node_coord):
        y_pred, std = self.gp.predict(node_coord, return_std=True)
        return y_pred, std

    def clear_observed_point(self):
        self.observed_points = []
        self.observed_value = []

    def update_gp(self):
        if self.observed_points:
            X = (np.array(self.observed_points)).reshape(-1, 2)
            # print(f"X is {X}")
            y = np.array(self.observed_value).reshape(-1,1)
            #print('X dimension:', len(X[:,0]))
            #print('y dimension:', len(y[:,0]))
            # print('X dimension:', X.shape, 'y dimension:', y.shape)
            ## judge dimensions of X and y
            if len(X[:,0])==len(y[:,0]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # print('update GP')
                    self.gp.fit(X, y)
            else:
                # print(f"len(observed_points) {len(X[:,0])} != len(observed_value) {len(y[:,0])}")
                raise ValueError(f"len(observed_points) {len(X[:,0])} != len(observed_value) {len(y[:,0])}")
                pass

    def update_node(self, node_coords):
        ###########################
        #print(self.node_coords.shape)
        y_pred, std = self.gp.predict(node_coords, return_std=True)

        return y_pred, std

    def evaluate_RMSE(self, y_true):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)
        RMSE = np.sqrt(mean_squared_error(y_pred, y_true))
        return RMSE

    def evaluate_F1score(self, y_true):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        score = self.gp.score(x1x2,y_true)
        return score

    def evaluate_cov_trace(self, X=None):
        if X is None:
            x1 = np.linspace(0, 1, 30)
            x2 = np.linspace(0, 1, 30)
            X = np.array(list(product(x1, x2)))
        _, std = self.gp.predict(X, return_std=True)
        trace = np.sum(std*std)
        return trace

    def evaluate_mutual_info(self, X=None):
        if X is None:
            x1 = np.linspace(0, 1, 30)
            x2 = np.linspace(0, 1, 30)
            X = np.array(list(product(x1, x2)))
        n_sample = X.shape[0]
        _, cov = self.gp.predict(X, return_cov=True)
        
        mi = (1 / 2) * np.log(np.linalg.det(0.01*cov.reshape(n_sample, n_sample) + np.identity(n_sample)))
        return mi

    def get_high_info_area(self): # t=0.4, beta=1):
        t = self.threshold
        beta = self.beta
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)
        
        high_measurement_area = []
        for i in range(900):
            if y_pred[i] + beta * std[i] >= t:
                high_measurement_area.append(x1x2[i])
        high_measurement_area = np.array(high_measurement_area)
        return high_measurement_area

    def plot(self, y_true):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)

        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)

        X0p, X1p = x1x2[:, 0].reshape(30, 30), x1x2[:, 1].reshape(30, 30)
        y_pred = np.reshape(y_pred, (30, 30))
        std = std.reshape(30,30)

        X = np.array(self.observed_points)

        fig = Figure(figsize=(15, 10))

        ax1 = fig.add_subplot(2, 3, 1)
        ax1.set_title('Predict mean')
        ax1.pcolormesh(X0p, X1p, y_pred, shading='auto', vmin=0, vmax=1, edgecolors="face")

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_title('Ground truth')
        ax2.pcolormesh(X0p, X1p, y_true.reshape(30, 30), shading='auto', vmin=0, vmax=1, edgecolors="face")

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title('Predict std')
        ax4.pcolormesh(X0p, X1p, std, shading='auto', vmin=0, vmax=1, edgecolors="face")

        return fig

if __name__ == '__main__':
    example = Gaussian2D()
    x1 = np.linspace(0, 1)
    x2 = np.linspace(0, 1)
    x1x2 = np.array(list(product(x1, x2)))
    y_true = example.distribution_function(X=x1x2)
    # print(y_true.shape)
    node_coords = np.random.uniform(0,1,(100,2))
    gp_ipp = GaussianProcessForIPP(node_coords)
    gp_ipp.plot(y_true.reshape(50,50))
    for i in range(node_coords.shape[0]):
        y_observe = example.distribution_function(node_coords[i].reshape(-1,2))
        # print(node_coords[i], y_observe)
        gp_ipp.add_observed_point(node_coords[i], y_observe)
        gp_ipp.update_gp()
        y_pre, std = gp_ipp.update_node()
        print(gp_ipp.evaluate_cov_trace())
    gp_ipp.plot(y_true)
    print(gp_ipp.evaluate_F1score(y_true))
    print(gp_ipp.gp.kernel_)