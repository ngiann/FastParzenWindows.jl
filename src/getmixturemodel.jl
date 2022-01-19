function getmixturemodel(Q, mu, C)

  MixtureModel([MvNormal(mu[kk,:],C[kk]) for kk=1:length(Q)], Q)

end
