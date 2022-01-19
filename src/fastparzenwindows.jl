fpw(X, r) = fastparzenwindows(X, r)

function fastparzenwindows(X, r)

  N = size(X, 1)

  D = size(X, 2)

  centres_ind = partition(X, r)

  Q, mu, C = softparzen(X, centres_ind, r)

  return Q, mu, C

end
