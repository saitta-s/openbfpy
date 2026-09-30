# Usage: julia --project=<validation environment> scripts/export_junction_reference.jl <output>
using DelimitedFiles
module Original
using LinearAlgebra, StaticArrays
include(joinpath(@__DIR__, "..", "openBF", "src", "vessel.jl"))
include(joinpath(@__DIR__, "..", "openBF", "src", "conjunctions.jl"))
include(joinpath(@__DIR__, "..", "openBF", "src", "bifurcations.jl"))
include(joinpath(@__DIR__, "..", "openBF", "src", "anastomosis.jl"))
end

output = abspath(ARGS[1]); mkpath(output)
blood = Original.Blood(Dict("mu"=>.004, "rho"=>1060.))
function vessel(i)
    config = Dict{Any,Any}("label"=>"v$i", "sn"=>i, "tn"=>i+1,
        "L"=>.01, "M"=>10, "E"=>700000., "R0"=>.003, "h0"=>.0003)
    Original.Vessel(config, blood, 10, ["P"])
end
v1, v2, v3 = vessel(1), vessel(2), vessel(3)
for (tag, n, signs) in (("conjunction", 2, [1.,-1.]), ("bifurcation", 3, [1.,-1.,-1.]),
                         ("anastomosis", 3, [1.,1.,-1.]))
    U = vcat(collect(range(.1, .3, length=n)), collect(range(.06, .08, length=n)))
    k = collect(range(40., 50., length=n)); W = ones(n)
    if n == 2
        F = Original.getFconj(v1, v2, U, k, W, blood.rho)
        J = Original.getJconj(v1, v2, U, k, blood.rho)
    elseif tag == "bifurcation"
        F = Original.getF(v1, v2, v3, U, k, W)
        J = Original.getJbif(v1, v2, v3, U, k)
    else
        F = Original.getFan(v1, v2, v3, U, k, W)
        J = Original.getJan(v1, v2, v3, U, k)
    end
    writedlm(joinpath(output, tag*"_residual.csv"), F, ',')
    writedlm(joinpath(output, tag*"_jacobian.csv"), J, ',')
end
println("Exported actual upstream junction residuals and Jacobians")
