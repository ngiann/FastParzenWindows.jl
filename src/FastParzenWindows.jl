module FastParzenWindows

    using StatsFuns, Random, Printf, SparseArrays, LinearAlgebra, Distributions, MLBase

    include("fastparzenwindows.jl")

    include("softparzen.jl")

    include("partition.jl")

    include("spiraldata.jl")

    include("getmixturemodel.jl")

    include("cvfpw.jl")


    export fpw, spiraldata, getmixturemodel, cv_fpw

end
