# Export the actual upstream MUSCL kernel without third-party dependencies.
# Usage: julia scripts/export_kernel_reference.jl output_directory
using DelimitedFiles

module Original
using LinearAlgebra
include(joinpath(@__DIR__, "..", "openBF", "src", "vessel.jl"))
# Only muscl! is executed. Network-dependent functions are parsed but not called.
struct Network end
include(joinpath(@__DIR__, "..", "openBF", "src", "solver.jl"))
end

output = abspath(ARGS[1])
mkpath(output)
for viscous in (false, true)
    config = Dict{Any,Any}("label"=>"reference", "sn"=>1, "tn"=>2,
        "L"=>0.01, "M"=>10, "E"=>700000., "R0"=>0.003, "h0"=>0.0003,
        "gamma_profile"=>2, "visco-elastic"=>viscous)
    blood = Original.Blood(Dict("mu"=>0.004, "rho"=>1060.))
    v = Original.Vessel(config, blood, 10, ["A", "Q", "u", "P"])
    for i in 1:v.M
        v.A[i] *= 1 + 0.02*sin(2*pi*(i-1)/(v.M-1))
        v.Q[i] = 1e-6*(1 + 0.1*cos(2*pi*(i-1)/(v.M-1)))
        v.u[i] = v.Q[i]/v.A[i]
    end
    v.U00A, v.UM1A = v.A[1], v.A[end]
    v.U00Q, v.UM1Q = v.Q[1], v.Q[end]
    tag = viscous ? "viscoelastic" : "elastic"
    writedlm(joinpath(output, tag*"_input.csv"), hcat(v.A, v.Q), ',')
    writedlm(joinpath(output, tag*"_geometry.csv"), hcat(v.A0, v.beta, v.gamma[2:end-1], v.Cv), ',')
    Original.muscl!(v, 1e-5, blood)
    writedlm(joinpath(output, tag*"_output.csv"), hcat(v.A, v.Q, v.u), ',')
end
println("Exported upstream kernel fixtures to ", output)
