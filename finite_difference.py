import market
import numpy as np
import matplotlib.pyplot as plt


class FiniteDifferenceGrid:

    def __init__(self, option, asset, rf, asset_step):
        self._validate_input(asset_step, option)
        self._nas = 2 * int(option.strike / asset_step)
        self._asset_step = asset_step
        self._nts = GridConfigurator.get_nts(asset.volatility,
                                             option.expiration,
                                             self._nas)
        self._time_step = option.expiration / self._nts
        self._bond = market.Bond(option.calculate_payoff(0), rf)
        self._abc = GridConfigurator.get_coefficients(self._nas,
                                                      self._time_step,
                                                      asset.volatility,
                                                      rf)
        self.grid = np.array(
            [[option.calculate_payoff(asset_step * i) for i in range(self._nas + 1)]],
            dtype=np.float64
        )

    @staticmethod
    def _validate_input(asset_step, option):
        assert (option.strike / asset_step) % 1 == 0, 'Approximation of infinity must be a multiple of asset step'

    def get_axes(self):
        return np.array([
            [self._time_step * k for k in range(self._nts + 1)],
            [self._asset_step * i for i in range(self._nas + 1)]
        ])

    def generate(self):
        while self._has_next():
            self.grid = np.append(self.grid, self._next_step(), axis=0)
        self.grid = np.flipud(self.grid)

    def _has_next(self):
        return np.shape(self.grid)[0] < self._nts + 1

    def _next_step(self):
        next_ = self._get_regular_points_from(self._get_current_step())
        next_[0] = self._get_lower_bound()
        next_[-1] = self._get_upper_bound(next_)
        return [next_]

    def _get_current_step(self):
        return self.grid[-1]

    def _get_regular_points_from(self, current):
        next_ = np.zeros_like(current)
        for i in np.arange(1, self._nas, 1):
            next_[i] = self._get_single_point(i, current)
        return next_

    def _get_lower_bound(self):
        return self._bond.discount(self._time_step, np.shape(self.grid)[0])

    @staticmethod
    def _get_upper_bound(next_):
        return 2 * next_[-2] - next_[-3]

    def _get_single_point(self, asset_step_id, current):
        return self._abc[0, asset_step_id] * current[asset_step_id - 1] + \
               (1 + self._abc[1, asset_step_id]) * current[asset_step_id] + \
               self._abc[2, asset_step_id] * current[asset_step_id + 1]


class GridConfigurator:

    @staticmethod
    def get_nts(sigma, expiration, nas):
        dt = .9 / (sigma * nas) ** 2  # for stability
        return int(expiration / dt) + 1  # integer number of time steps

    @staticmethod
    def get_coefficients(nas, time_step, sigma, rf):
        return np.array([
            [.5 * ((sigma * i) ** 2 - rf * i) * time_step for i in range(nas + 1)],
            [-((sigma * i) ** 2 + rf) * time_step for i in range(nas + 1)],
            [.5 * ((sigma * i) ** 2 + rf * i) * time_step for i in range(nas + 1)]
        ])


class Plotter:

    def __init__(self, values, axes):
        self._values = values
        self._x_axis = axes[0]
        self._y_axis = axes[1]

    def surface_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(self._x_axis, self._y_axis)
        ax.plot_surface(x, y, self._values,
                        cmap='coolwarm')
        ax.set_xlabel('Time')
        ax.set_ylabel('Asset')
        ax.set_zlabel('Option Price')
        plt.show()
