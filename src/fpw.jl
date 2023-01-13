"""
    fpw(X, r; gamma = 1e-6, seed = 1)

ntrols the random number generator used for initialising the starting point of the optimisation.

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
