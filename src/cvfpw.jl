"""
    cvresults =  cv_fpw(X, r_range; numFolds = 10, seed=1)

Performs cross validation for the radius parameter.
Candidate values for the radius parameter are specified in `r_range`.
Default number of folds is 10.
Seed controls the generation of the folds.

# Example
```julia-repl
julia> using Statistics, PyPlot
julia> X = spiraldata(300)
julia> r_range = LinRange(0.01, 2.0, 100)
julia> cvresults =  cv_fpw(X, r_range)
julia> r_perf = mean(cvresults, dims=2)
julia> best_index = argmax(r_perf)
julia> r_best = r_range[best_index]
julia> mix = fpw(X, r_best)
julia> x = rand(mix, 1000)'
julia> plot(X[:,1], X[:,2], "bo", label="data")
julia> plot(x[:,1], x[:,2], ".r", label="generated", alpha=0.7)
julia> legend()
```
"""
function cv_fpw(X, r_range; numFolds = 10, seed=1)


  Random.seed!(seed)

  randRepeats =  7

  N = size(X, 1)
  D = size(X, 2)

  # partition datasets into disjoint sets
  sets = collect(Kfold(N, numFolds))

  score = zeros(length(r_range), numFolds)

  for (r_index, r) in enumerate(r_range)

    @printf("=> r = %f ",r)

    for i = 1:numFolds

      # define indices for training set
      trainInd = sets[i]
      Xtrain   = @view X[trainInd,:]

      # define indices for testing set
      testInd = setdiff(collect(1:N), trainInd)
      Xtest   = @view X[testInd,:]

      for repeat=1:randRepeats

        # get Parzen window
        mixturemodel = fpw(Xtrain, r)

        # get log-likelihood on test set
        logpdfmix(x) = logpdf(mixturemodel, x)

        for nn=1:length(testInd)
          @inbounds score[r_index, i] += logpdfmix(@view Xtest[nn,:]) / randRepeats
        end

      end

      @printf(".")

    end
    @printf("\n")
  end


  return score

end
