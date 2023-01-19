"""

  p = fpw(X, r; gamma = 1e-6, seed = 1)

Estimate density through the fast parzen windows density algorithm. 
The algorithm partitions the data space in hyperdiscs of radius `r`.
Data items in matrix `X` are then 'softly' assigned to the partitions.
The local density in each partition is modelled by a Gaussian distribution.
The global density estimate is returned as a Gaussian mixture model of type `Distributions.MixtureModel`.

The implementation is based on **X. Wang, P. Tino, M. A. Fardal, S. Raychaudhury and A. Babul, "Fast parzen window density estimator," 2009 International Joint Conference on Neural Networks, 2009, pp. 3267-3274.**

# Parameters

* `X` is a N×D data matrix, i.e. there are N data items of dimension D.
* `r` is a scalar specifying the common radius of the hyperdiscs
* `seed` controls the random number generator that randomly picks data items as hyperdisc centres.
* `gamma` is a scalar that specified a multiple of the identity matrix, i.e. γI, added to the covariance matrices of the local Gaussian for numerical stability.

# Returns

* `p` a Gaussian mixture model as type `Distributions.MixtureModel`

# Example
```julia-repl
julia> X = spiraldata(300)
julia> mix = fpw(X, 0.05)
julia> x = rand(mix, 1000)'
julia> plot(X[:,1], X[:,2], "bo", label="data")
julia> plot(x[:,1], x[:,2], ".r", label="generated", alpha=0.7)
julia> legend()
```
"""
fpw(X, r; gamma = 1e-6, seed = 1) = fastparzenwindows(X, r, gamma, seed)

function fastparzenwindows(X, r, gamma, seed)

  centres_ind = partition(X, r, seed)

  Q, mu, C = softparzen(X, centres_ind, r, gamma)

  getmixturemodel(Q, mu, C)

end
