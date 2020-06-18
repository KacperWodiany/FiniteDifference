import random_walk as rw


class Option:

    def __init__(self, expiration, strike, payoff, lower_barrier=None, upper_barrier=None):
        self.expiration = expiration
        self.strike = strike
        self._payoff = payoff
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier

    def calculate_payoff(self, asset_price):
        if self.lower_barrier is not None and asset_price <= self.lower_barrier:
            return 0
        elif self.upper_barrier is not None and asset_price >= self.upper_barrier:
            return 0
        else:
            return self._payoff(asset_price, self.strike)


class Bond:

    def __init__(self, face_value, rf):
        self._face_value = face_value
        self.rf = rf

    def discount(self, time_step, n=1):
        return self._face_value * (1 - self.rf * time_step) ** n


class Asset:

    def __init__(self, drift, volatility):
        self.drift = drift
        self.volatility = volatility

    def random_path(self, initial_price, time_horizon, time_step):
        bm = rw.BrownianMotion(time_horizon, time_step)
        return bm.generate_exponential(initial_price,
                                       self.drift,
                                       self.volatility)
