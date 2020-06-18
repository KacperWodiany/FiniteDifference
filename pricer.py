import argparse
import numpy as np
import finite_difference as fd
import market

payoffs = np.array([lambda s, e: max(s - e, 0), lambda s, e: max(e - s, 0)])


def run(option_type, strike, expiration, volatility, rf, ds, surface, timezero):
    stock = market.Asset(0, volatility)
    option = market.Option(expiration, strike, payoffs[option_type])
    finite_diff = fd.FiniteDifference(option, stock, rf, ds)
    grid = finite_diff.generate_grid()
    axes = finite_diff.get_grid_axes()
    plotter = fd.Plotter(grid, axes[0], axes[1])

    if surface:
        plotter.surface()

    if timezero:
        plotter.time_zero_price()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pricing options with Explicit Finite Difference method.")
    parser.add_argument('-type', '--option-type', type=int, metavar='', default=0,
                        help='Available options: 0-european call, 1-european put')
    parser.add_argument('-s', '--strike', type=int, metavar='', required=True, help='Exercise price of an option')
    parser.add_argument('-e', '--expiration', type=float, metavar='', default=1,
                        help='Expiration of an option (as a fraction of a year)')
    parser.add_argument('-v', '--volatility', type=float, metavar='', required=True,
                        help='Volatility of an underlying asset')
    # parser.add_argument('-m', '--drift', metavar='', help='Drift of an asset')
    parser.add_argument('-rf', '--risk-free', type=float, metavar='', required=True, help='Risk free rate')
    parser.add_argument('-dS', '--asset-step', type=int, metavar='', default=10,
                        help='Asset step on Finite Difference grid. Small values (<10) can extend computation time')
    parser.add_argument('--surface', help='Plot 3D option price surface', action='store_true')
    parser.add_argument('--timezero', help='Plot asset price vs option price at time 0', action='store_true')

    args = parser.parse_args()

    print(args)

    run(args.option_type, args.strike, args.expiration, args.volatility,
        args.risk_free, args.asset_step, args.surface, args.timezero)
