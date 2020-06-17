import market
import numpy as np


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
        self._grid = np.array(
            [[option.calculate_payoff(asset_step * i) for i in range(self._nas + 1)]])

    @staticmethod
    def _validate_input(asset_step, option):
        assert (option.strike / asset_step) % 1 == 0, 'Approximation of infinity must be a multiple of asset step'

    def generate(self):
        while self._has_next():
            self._grid = np.append(self._grid, self._next_step(), axis=0)
        self._grid = np.flipud(self._grid)

    def _has_next(self):
        return np.shape(self._grid)[0] < self._nts + 1

    def _next_step(self):
        next_ = self._get_regular_points_from(self._get_current_step())
        next_[0] = self._get_lower_bound()
        next_[-1] = self._get_upper_bound(next_)
        return [next_]

    def _get_current_step(self):
        return self._grid[-1, :]

    def _get_regular_points_from(self, current):
        next_ = np.zeros_like(current)
        for i in np.arange(1, self._nas, 1):
            next_[i] = self._get_single_point(i, current)
        return next_

    def _get_lower_bound(self):
        return self._bond.discount(self._time_step, np.shape(self._grid)[0])

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
