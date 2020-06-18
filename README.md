# FiniteDifference
Implementation of Explicit Finite Difference method for option pricing.

Provides tools for pricing customly defined options. Program allows early-exercise option and knock-out options with upper or lower barrier (or both).
One can define his own payoff by lambda expression of asset price and strike. 

To generate grid of option values, instance of FiniteDifference class need to be created.
To create custom option use Option class. Use Asset class to define an underlying asset. Use those 2 object and some other parameters to initialize FiniteDifference object.
Invoke generate_grid() method, to generate grid of option prices. Grid is stored in object and can be accessed by get_grid(). 
Due to stability reasons user cannot define his own time step. Setting asset step less than 10 can result in long generating time.

Plotter provides easy way to plot surfaces of option values and curves of prices at time 0.

To get option value for time or asset price, that is not multiple of defined asset step, use Interpolator class. 
It provides bilinear interpolation tool. Initializing with grid and scales of time and asset, one can get value at custom point, using value_at(x,y).
Using Interpolator one can price options for whole paths.

Simple CLI is also provided.
