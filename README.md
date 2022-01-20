# FastParzenWindows.jl

Implementation of the Fast Parzen Window Density Estimator described in 

*X. Wang, P. Tino, M. A. Fardal, S. Raychaudhury and A. Babul, "Fast parzen window density estimator," 2009 International Joint Conference on Neural Networks, 2009, pp. 3267-3274.*

The algorithm presented in the paper has two versions called 'hard' and 'soft'.  This repository only provides the 'soft' version.

# Brief description

This is a technique for estimating a probability density from an observed set of data points. The data space is partitioned in hyper-discs of fixed radii `r` and each partition is modelled with a Gaussian density. The method is non-parametric in the sense that it automatically decides on the number of Gaussian densities it needs. The final model is a mixture of Gaussians with each Gaussian fitted locally to a partition.

# How to use

There are two functions of interest: `fpw` and `cv_fpw`.

- `fpw` takes two arguments, a N×D data matrix `X` and a scalar `r` which expresses the radius of the hyper-discs in which the data space is partitioned. The output is an object of the type `Distributions.MixtureModel`.
- `cv_fpw` takes two arguments, a N×D data matrix `X` and a range of candidate radii of the hyper-discs. It performs cross-validation for each candidate `r` and returns a matrix of out-of-sample log-likelihoods of dimensions (number of `r` candidates)×(number of folds).



# Example

We use a dataset taken from the paper. We generate 300 data points using:
```
X = spiraldata(300)

using PyPlot # must be independently installed. Of course any other plotting package can be used instead.
plot(X[:,1], X[:,2], "o")
```


