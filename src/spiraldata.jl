"""
    X = spiraldata(N)

Generates data points on a 2D spiral returned as a N×2 matrix X.
"""
function spiraldata(N, seed=1)

    rg = MersenneTwister(seed)

    t = rand(rg, Uniform(3, 15), N)

    x1 = 0.04 * t .* sin.(t) .+ randn(rg, N)*0.01

    x2 = 0.04 * t .* cos.(t) .+ randn(rg, N)*0.01

    [x1 x2]

end
