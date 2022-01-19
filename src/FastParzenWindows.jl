module FastParzenWindows

    using StatsFuns, Random, Printf, SparseArrays, LinearAlgebra, Distributions, MLBase

    include("fastparzenwindows.jl")

    include("softparzen.jl")

    include("partition.jl")

    include("spiraldata.jl")

    include("getmixturemodel.jl")

    include("cvfpw.jl")


    export fpw, spiraldata, cv_fpw

end
