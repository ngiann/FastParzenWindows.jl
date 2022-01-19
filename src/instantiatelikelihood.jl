#####################################################################
function instantiatelikelihood(p, mu, C)
#####################################################################

  D = length(mu[1,:])
  M = length(p)
  @assert(abs(sum(p)-1.0)<1e-6) # make sure coeff sum up to one

  # define components of mixture
  component = [MvNormal(vec(mu[mm,:]), C[mm]) for mm in 1:M]
  

  # define mixture likelihood function
  function pdfMixture(x)
    likel = 0.0
    for mm=1:M
      @inbounds likel += p[mm] * pdf(component[mm], x)
    end
    return likel
  end

  # define mixture log-likelihood function
  function logpdfMixture(x)
    loglikel = zeros(M)
    for mm=1:M
      @inbounds loglikel[mm] = log(p[mm]) + logpdf(component[mm], x)
    end
    return logsumexp(loglikel)
  end

  # define mixture likelihood function
  c = Categorical(p)
  function randMixture()
    index = rand(c)
    rand(component[index])
  end

  return pdfMixture, logpdfMixture, randMixture

end
