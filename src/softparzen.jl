##############################################################
function softparzen(X, S_centre, r, γ = 1e-6)
##############################################################

  N, D = size(X)
  M = length(S_centre)

  # Usually kernel has also some normalisation constant
  # but in the following calculations it cancels out anyway
  K = zeros(M,N)
  for j in 1:M
    for n in 1:N
      @inbounds K[j,n] = exp(-0.5*norm(X[S_centre[j],:] - X[n,:])^2 / (r*r))
    end
  end

  # Soft version for coefficients
  Q = zeros(M)
  sumK = sum(K)
  for j in 1:M
    @inbounds Q[j] = sum(K[j,:]) / sumK
  end


  # responsibilities - note different to standard mixture models
  # See equation (6)
  resp = zeros(M,N)
  for j in 1:M
    sum_Kj_over_n = sum(K[j,:])
    for n in 1:N
      @inbounds resp[j,n] = K[j,n] / sum_Kj_over_n
    end
  end


  # Soft version for means
  # See equation (7)
  mu = zeros(M, D)
  for j=1:M
    for n=1:N
      for i=1:D
        @inbounds mu[j,i] += resp[j,n] * X[n,i]
      end
    end
  end


  # Soft version for covariances
  C = [zeros(D, D) for j in 1:M]

  for j in 1:M
    for i in 1:D
      for d in 1:D
        for n in 1:N
          @inbounds C[j][i,d] += resp[j,n] * ((X[n,i] - mu[j,i]) * (X[n,d] - mu[j,d]))
        end
      end
    end

    C[j] = (C[j] + C[j]')*0.5 + γ*I

  end

  return Q, mu, C

end
