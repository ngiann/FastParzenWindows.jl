# FastParzenWindows.jl

Implementation of the Fast Parzen Window Density Estimator described in 

*X. Wang, P. Tino, M. A. Fardal, S. Raychaudhury and A. Babul, "Fast parzen window density estimator," 2009 International Joint Conference on Neural Networks, 2009, pp. 3267-3274.*

The algorithm presented in the paper has two versions called 'hard' and 'soft'.  This repository only provides the 'soft' version.

# Brief description

This is a non-parametric probability density estimation approach. It covers the data space with fixed radii hyper-balls with densities represented by full covariance Gaussians

# How to use

There are two functions of interest: `fpw` and `cv_fpw`.

# Example
