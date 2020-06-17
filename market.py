import random_walk as rw


class Option:

    def __init__(self, expiration, strike, payoff):
        self.expiration = expiration
        self.strike = strike
        self._payoff = payoff

    def calculate_payoff(self, asset_price):
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
