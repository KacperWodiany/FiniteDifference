import argparse
import numpy as np
import finite_difference as fd
import market

payoffs = np.array([lambda s, e: max(s - e, 0), lambda s, e: max(e - s, 0)])


def run(option_type, strike, expiration, volatility, rf,
        ds, early_ex, upper_bar, lower_bar, surface, timezero):
    stock = market.Asset(0, volatility)
    option = market.Option(expiration, strike, payoffs[option_type],
                           upper_barrier=upper_bar, lower_barrier=lower_bar)
    finite_diff = fd.FiniteDifference(option, stock, rf, ds, early_exercise=early_ex)
    finite_diff.generate_grid()
    grid = finite_diff.get_grid()
    axes = finite_diff.get_grid_axes()
    plotter = fd.Plotter(grid, axes[0], axes[1])

    if surface:
        plotter.surface()

    if timezero:
        plotter.time_zero()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pricing options with Explicit Finite Difference method")
    parser.add_argument('-type', '--option-type', type=int, default=0,
                        help='Available options: 0-call, 1-put')
    parser.add_argument('-s', '--strike', type=int, required=True, help='Exercise price of an option')
    parser.add_argument('-e', '--expiration', type=float, default=1,
                        help='Expiration of an option (as a fraction of a year)')
    parser.add_argument('-v', '--volatility', type=float, required=True,
                        help='Volatility of an underlying asset')
    parser.add_argument('-rf', '--risk-free', type=float, required=True, help='Risk free rate')
    parser.add_argument('-dS', '--asset-step', type=int, default=10,
                        help='Asset step on Finite Difference grid. Small values (<10) can extend computation time')
    parser.add_argument('-ex', '--early-exercise', type=bool, default=False,
                        help='True if option can be exercised before maturity (American options)')
    parser.add_argument('-lb', '--lower-barrier', type=int, default=None,
                        help='Lower barrier level')
    parser.add_argument('-ub', '--upper-barrier', type=int, default=None,
                        help='Upper barrier level')
    parser.add_argument('--surface', help='Plot 3D option price surface', action='store_true')
    parser.add_argument('--timezero', help='Plot asset price vs option price at time 0', action='store_true')

    args = parser.parse_args()

    run(args.option_type, args.strike, args.expiration, args.volatility,
        args.risk_free, args.asset_step, args.early_exercise,
        args.upper_barrier, args.lower_barrier, args.surface, args.timezero)
