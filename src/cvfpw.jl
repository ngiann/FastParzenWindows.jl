"""
    cvresults = cv_fpw(X, r_range; numfolds = 10, seed = 1, gamma = 1e-6, randrepeats = 7)

Performs cross validation for the radius parameter.
Candidate values for the radius parameter are specified in `r_range`.
Seed controls the generation of the folds.
Returns the a matrix of dimensions (number of radii candidates)×(number of folds) of log-likelihoods evaluated on left out folds.

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
function cv_fpw(X, r_range; numfolds = 10, seed = 1, gamma = 1e-6, randrepeats = 7)


  Random.seed!(seed)

  # dimensions of data

  N, D = size(X)

  # partition datasets into disjoint sets

  sets = collect(Kfold(N, numfolds))

  score = zeros(length(r_range), numfolds)

  progressbar = Progress(length(r_range))

  for (r_index, r) in enumerate(r_range)

    for i = 1:numfolds

      # define indices for training set
      trainInd = sets[i]
      Xtrain   = @view X[trainInd,:]

      # define indices for testing set
      testInd = setdiff(collect(1:N), trainInd)
      Xtest   = @view X[testInd,:]

      for repeat in 1:randrepeats

        # get Parzen window
        mixturemodel = fpw(Xtrain, r; gamma = gamma, seed = seed + repeat)

        # get log-likelihood on test set
        logpdfmix(x) = logpdf(mixturemodel, x)

        for n in 1:length(testInd)
        
          @inbounds score[r_index, i] += logpdfmix(@view Xtest[n,:]) / randrepeats
        
        end

      end

    end

    ProgressMeter.next!(progressbar; showvalues = [(:r,r)])

  end


  return score

end
