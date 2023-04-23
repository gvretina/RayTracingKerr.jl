module image

import ..angles
import ..spacetime

using OrdinaryDiffEq
using LinearAlgebra
using StaticArrays
using Colors
using Images, ImageView

function get_background(filename::String)
    background = load(filename)
    h, w = size(background)
    th_b = LinRange(0, h, h)
    ph_b = LinRange(0, w, w)
    th_b = th_b * (pi / h)
    ph_b = pi .- ph_b * (2 * pi / (w)) #pi/2 .- ph_b*(pi/(w))
    back_angs = [[th_b[i], ph_b[j]] for i in 1:h, j in 1:w]
    background, back_angs
end


function update_image!(img::Array{RGBA{N0f8},2},
    th::Vector{T}, ph::Vector{T},
    g::SMatrix{4,4,Float64,16}, gu::SMatrix{4,4,Float64,16},
    tspan::Tuple{T,T}, tol::Real,
    pos::Vector{T}, R_euler::AbstractArray{T},
    background::Array{RGB{N0f8},2}, back_angs::Matrix{Vector{Float64}},
    metric::Metric
) where {Metric,T<:Real}

    condition(s, λ, integrator) = (3 < norm(s[2:D]) < 4) && (abs(angles.cart2polar(s[2:D])[1] - pi / 2) < 1e-2)
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)
    postemp = @view pos[2:end]
    init_r = norm(postemp)
    h, w = size(background)


    N = length(th)
    M = length(ph)
    @fastmath @inbounds Threads.@threads for k in CartesianIndices((N, M))
        i, j = Tuple(k)
        normal = T[0, -1*sign(pos[2])*1, 0, 0]
        fov = T[0, sin(ph[j]), sin(th[i])]
        ##### Rotate normal Velocities appropriately ########
        normal[2:end] = R_euler * (normal[2:end] + fov)

        u = angles.get_final_velocity(normal, g, gu)

        prob = ODEProblem(spacetime.geodesic, SVector{2D,T}(vcat(pos, u)), tspan, metric)

        sims = solve(prob, AutoTsit5(Rosenbrock23(autodiff=false)), isoutofdomain=(y, p, t) -> ((x -> x > init_r * 3)(norm(y[2:D])) && any(isnan, y)),
            callback=cb, abstol=tol, reltol=tol, #unstable_check=(y,p,t)->any(isnan,y),
            maxiters=1e4, verbose=false)#, dtmin =1e-12, force_dtmin=true)

        xyz = [sims.u[i][2:D] for i in 1:length(sims.t)]
        fin_r = norm(xyz[end])

        if fin_r > init_r
            angs = angles.cart2polar(xyz[end])
            norms = [norm(back_angs[k, l] - angs, Inf) for k in 1:h, l in 1:w]
            #norms = [haversine(back_angs[i,j],angs) for i in 1:h, j in 1:w]
            idx = argmin(norms)
            img[i, j] = background[idx]

        elseif 3 < fin_r < 4
            img[i, j] = colorant"white"
        else
            img[i, j] = colorant"black"
        end

        next!(p)
    end
    finish!(p)
end



end
