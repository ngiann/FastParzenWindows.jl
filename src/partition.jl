###################################################
function partition(X, r_threshold, seed = 1)
###################################################

  rg = MersenneTwister(seed)

  N = size(X, 1)
  D = size(X, 2)

  S_centre = spzeros(N)  # for storing indices of data items that become centres
  T_used   = BitArray(undef, N) # for marking data items as used
  fill!(T_used, false)   # mark all data items initially as unused

  # select first data point
  index           = ceil(Int, rand(rg)*N)
  T_used[index]   = 1  # mark it used
  S_centre[index] = 1  # mark it as centre

  while sum(T_used)<N # keep processing while there are unused items

    # randomly choose an unused data item
    unused_indices = findall(T_used .== 0)
    unused_index   = unused_indices[ceil(Int, rand(rg)*length(unused_indices))]

    # mark it immediately as used
    T_used[unused_index] = 1

    # check whether it falls inside the radius of existing centres
    INSIDE = 0
    for centre_index in S_centre.nzind

      # calculate Euclidean distance
      EuclDist = norm(X[centre_index,:] - X[unused_index,:])
      if EuclDist <= r_threshold
        INSIDE = 1
        break
      end

    end

    # if inside is still 0, mark it a centre
    if INSIDE == 0
      S_centre[unused_index] = 1
    end

  end

  return S_centre.nzind

end
