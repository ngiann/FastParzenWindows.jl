module FastParzenWindows

    using ProgressMeter, StatsFuns, Random, Printf, SparseArrays, LinearAlgebra, Distributions, MLBase

    include("fpw.jl")

    include("softparzen.jl")

    include("partition.jl")

    include("spiraldata.jl")

    include("getmixturemodel.jl")

    include("cvfpw.jl")


    export fpw, spiraldata, cv_fpw

end
