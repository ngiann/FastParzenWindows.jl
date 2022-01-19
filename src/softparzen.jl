##############################################################
function softparzen(X, S_centre, r)
##############################################################

  N = size(X, 1)
  D = size(X, 2)
  M = length(S_centre)

  # Usually kernel has also some normalisation constant
  # but in the following calculations it cancels out anyway
  K = zeros(M,N)
  for mm=1:M
    for nn=1:N
      @inbounds K[mm,nn] = exp(-0.5*norm(X[S_centre[mm],:] - X[nn,:])^2 / (r*r))
    end
  end

  # Soft version for coefficients
  Q = zeros(M)
  sumK = sum(K)
  for mm=1:M
    @inbounds Q[mm] = sum(K[mm,:]) / sumK
  end


  # normalise
  resp = zeros(M,N)
  for mm=1:M
    sum_Km_over_n = sum(K[mm,:])
    for nn=1:N
      @inbounds resp[mm,nn] = K[mm,nn] / sum_Km_over_n
    end
  end

  # Soft version for means
  mu = zeros(M, D)
  for mm=1:M

    for nn=1:N
      for ii=1:D
        @inbounds mu[mm,ii] += resp[mm,nn]*X[nn,ii]
      end
    end

  end

  # Soft version for covariances
  C = [zeros(D, D) for mm=1:M]
  for mm=1:M

    for ii=1:D
      for jj=1:D
        for nn=1:N
          @inbounds C[mm][ii,jj] += resp[mm,nn] * ((X[nn,ii] - mu[mm,ii]) * (X[nn,jj] - mu[mm,jj]))
        end
      end
    end

    C[mm] = (C[mm] + C[mm]')*0.5 + 1e-6*I
  end

  return Q, mu, C

end
