######################################################
function cv_fpw(X, r_range, seed=1)
######################################################

  Random.seed!(seed)

  randRepeats =  7
  numFolds    = 10

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
