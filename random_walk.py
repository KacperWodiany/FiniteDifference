import numpy as np
import math


class BrownianMotion:

    def __init__(self, time_horizon, time_step):
        BrownianMotion._validate_input(time_horizon, time_step)
        self._time_horizon = time_horizon
        self._time_step = time_step

    @staticmethod
    def _validate_input(time_horizon, time_step):
        assert math.isclose(time_horizon % time_step, 0, abs_tol=1e-16), \
            'Time horizon must be a multiple of time step'

    def generate(self):
        increments = np.random.normal(0,
                                      self._time_step,
                                      int(self._time_horizon / self._time_step))
        return np.cumsum(np.concatenate(([0], increments)))

    def generate_exponential(self, initial_value, mu, sigma):
        brownian_motion = self.generate()
        time_steps = self._get_time_steps()
        return initial_value * np.exp(
            (mu - .5 * sigma ** 2) * time_steps + sigma * brownian_motion)

    def _get_time_steps(self):
        return np.concatenate(
            (np.arange(0, self._time_horizon, self._time_step), [1]))
