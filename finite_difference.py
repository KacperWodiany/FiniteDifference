import market
import numpy as np
import matplotlib.pyplot as plt
import math


class FiniteDifference:

    def __init__(self, option, asset, rf, asset_step, lower_barrier=None, upper_barrier=None, early_exercise=False):
        self._validate_input(asset_step, option, lower_barrier, upper_barrier)
        self._nas = 2 * int(option.strike / asset_step)
        self._asset_step = asset_step
        self._nts = GridConfigurator.get_nts(asset.volatility,
                                             option.expiration,
                                             self._nas)
        self._time_step = option.expiration / self._nts
        self._lower_barrier_step = int(lower_barrier / asset_step) if lower_barrier is not None else 0
        self._upper_barrier_step = int(upper_barrier / asset_step) if upper_barrier is not None else self._nas
        self._early = early_exercise
        self._bond = market.Bond(option.calculate_payoff(0), rf)
        self._abc = GridConfigurator.get_coefficients(self._nas,
                                                      self._time_step,
                                                      asset.volatility,
                                                      rf)
        self._grid = np.array(
            [[option.calculate_payoff(asset_step * i) for i in range(self._nas + 1)]],
            dtype=np.float64
        )



    @staticmethod
    def _validate_input(asset_step, option, lower_barrier, upper_barrier):
        assert math.isclose(option.strike % asset_step, 0, abs_tol=1e-16),\
            'Strike must be a multiple of asset step'
        if lower_barrier is not None:
            assert math.isclose(lower_barrier % asset_step, 0, abs_tol=1e-16),\
                'Lower barrier must be a multiple of asset step'
        if upper_barrier is not None:
            assert math.isclose(upper_barrier % asset_step, 0, abs_tol=1e-16), \
                'Lower barrier must be a multiple of asset step'

    def get_grid_axes(self):
        return np.array([
            self._time_step * np.arange(0, self._nts + 1),
            self._asset_step * np.arange(0, self._nas + 1)
        ])

    def get_grid(self):
        return self._grid

    def generate_grid(self):
        while self._has_next():
            next_ = np.maximum(self._grid[0], self._next_step()) if self._early else self._next_step()
            self._grid = np.append(self._grid, next_, axis=0)
        self._grid = np.flipud(self._grid)
        return self._grid

    def _has_next(self):
        return np.shape(self._grid)[0] < self._nts + 1

    def _next_step(self):
        next_ = self._get_regular_points_from(self._get_current_step())
        next_[0:self._lower_barrier_step + 1] = self._get_lower_bound()
        if not self._upper_barrier_step == self._nas:
            next_[self._upper_barrier_step - 1:-1] = self._get_upper_bound(next_)
            next_[-1] = 0
        else:
            next_[-1] = self._get_upper_bound(next_)
        return [next_]

    def _get_current_step(self):
        return self._grid[-1]

    def _get_regular_points_from(self, current):
        next_ = np.zeros_like(current)
        for i in np.arange(max(self._lower_barrier_step, 1), min(self._upper_barrier_step, self._nas)):
            next_[i] = self._get_single_point(i, current)
        return next_

    def _get_single_point(self, asset_step_id, current):
        return self._abc[0, asset_step_id] * current[asset_step_id - 1] + \
               (1 + self._abc[1, asset_step_id]) * current[asset_step_id] + \
               self._abc[2, asset_step_id] * current[asset_step_id + 1]

    def _get_lower_bound(self):
        if self._lower_barrier_step == 0:
            return self._bond.discount(self._time_step, np.shape(self._grid)[0])
        else:
            return np.repeat(0, self._lower_barrier_step + 1)

    def _get_upper_bound(self, next_):
        if self._upper_barrier_step == self._nas:
            return 2 * next_[-2] - next_[-3]
        else:
            return np.repeat(0, self._nas - self._upper_barrier_step + 1)


class GridConfigurator:

    @staticmethod
    def get_nts(sigma, expiration, nas):
        dt = .9 / (sigma * nas) ** 2  # for stability
        return int(expiration / dt) + 1  # integer number of time steps

    @staticmethod
    def get_coefficients(nas, time_step, sigma, rf):
        i = np.arange(0, nas + 1)
        return np.array([
            .5 * ((sigma * i) ** 2 - rf * i) * time_step,
            -((sigma * i) ** 2 + rf) * time_step,
            .5 * ((sigma * i) ** 2 + rf * i) * time_step
        ])


class Interpolator:

    def __init__(self, grid, x_scale, y_scale):
        self._grid = grid
        self._x_scale = x_scale
        self._y_scale = y_scale

    def value_at(self, x, y):
        self._validate_input(x, y)
        corners_ids = self._find_corners_ids(self._find_initial_corner_id(x, y))
        corners_grid_values = self._values_in_corners(corners_ids)
        areas = self._get_rectangles_area(self._get_corners_scale_points(corners_ids),
                                          np.array([x, y]))
        return self._interpolate(areas, corners_grid_values)

    def _validate_input(self, x, y):
        assert np.min(self._x_scale) <= x <= np.max(self._x_scale), f'First coordinate out of grid range'
        assert np.min(self._y_scale) <= y <= np.max(self._y_scale), f'Second coordinate out of grid range'

    @staticmethod
    def _find_corners_ids(initial_corner):
        return np.array([
            initial_corner,
            [initial_corner[0] + 1, initial_corner[1]],
            [initial_corner[0] + 1, initial_corner[1] + 1],
            [initial_corner[0], initial_corner[1] + 1]
        ])

    def _find_initial_corner_id(self, x, y):
        x_id = self._adjust_id(self._x_scale, x)
        y_id = self._adjust_id(self._y_scale, y)
        return np.array([x_id, y_id])

    @staticmethod
    def _adjust_id(scale, value):
        id_ = np.argmax(scale >= value)
        if id_ == len(scale) - 1:
            return id_ - 1
        else:
            return id_

    def _values_in_corners(self, corners_ids):
        corners_ids = self._as_tuple(corners_ids)
        return np.array(
            [self._grid.item(corner_id) for corner_id in corners_ids]
        )

    @staticmethod
    def _as_tuple(array):
        return tuple(map(tuple, array))

    @staticmethod
    def _get_rectangles_area(corners, point):
        return np.prod(np.abs(corners - point), axis=1)

    def _get_corners_scale_points(self, corners_ids):
        t_corners_ids = corners_ids.transpose()
        t_corners_scale_points = np.array([
            self._x_scale[t_corners_ids[0]],
            self._y_scale[t_corners_ids[1]]
        ])
        return t_corners_scale_points.transpose()

    @staticmethod
    def _interpolate(areas, values):
        reordered_areas = areas[[2, 3, 0, 1]]
        return np.sum(reordered_areas * values) / np.sum(reordered_areas)

class Plotter:

    def __init__(self, values, x_axis, y_axis):
        self._values = values
        self._x_axis = x_axis
        self._y_axis = y_axis

    def surface(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(self._x_axis, self._y_axis)
        ax.plot_surface(x, y, self._values.transpose(),
                        cmap='coolwarm')
        ax.set_xlabel('Time')
        ax.set_ylabel('Asset')
        ax.set_zlabel('Option Price')
        plt.show()

    def time_zero_price(self):
        plt.plot(self._y_axis, self._values[0])
        plt.xlabel('Asset Price')
        plt.ylabel('Option Price')
        plt.title('Asset Price vs Option Price at time 0')
        plt.show()
