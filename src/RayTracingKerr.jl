module RayTracingKerr

using LinearAlgebra
using StaticArrays
using ForwardDiff
#using GLMakie
using OrdinaryDiffEq
using IterTools
import NaNMath; nm=NaNMath
using ProgressMeter
using Colors
using Images,ImageView
using Rotations

const D = 4
#T = Float64
function kerr_schild(xx::SVector{D,T})::SMatrix{D,D,T} where {T}
    M = 1

    t,x,y,z = xx
    #@assert !any(isnan, (t, x, y, z))

    # <https://en.wikipedia.org/wiki/Kerr_metric>
    η = @SMatrix T[a==b ? (a==1 ? -1 : 1) : 0 for a in 1:D, b in 1:D]
    a = 0.9                       # T(0.8)

    ρ = nm.sqrt(x^2 + y^2 + z^2)
    r = nm.sqrt(ρ^2 - a^2)/2 + nm.sqrt(a^2*z^2 + ((ρ^2 - a^2)/2)^2)
    f = 2*M*r^3 / (r^4 + a^2*z^2)
    k = SVector{D,T}(1,
                     (r*x + a*y) / (r^2 + a^2),
                     (r*y - a*x) / (r^2 + a^2),
                     z / r)

    g = @SMatrix T[η[a,b] + f * k[a] * k[b] for a in 1:D, b in 1:D]
    g
end

function dmetric(g::Metric, x::SVector{D,T}) where {Metric, T}
    dg = reshape(ForwardDiff.jacobian(g,x),(D,D,D))
    g, dg
end

function christoffel(metric::Metric, x::SVector{D,T}) where {Metric, T}
    g,dg = dmetric(metric, x)
    gu = inv(g(x))
    Γl = @SArray T[(dg[a,b,c] + dg[a,c,b] - dg[b,c,a]) / 2
                   for a in 1:D, b in 1:D, c in 1:D]
    @SArray T[gu[a,1] * Γl[1,b,c] +
              gu[a,2] * Γl[2,b,c] +
              gu[a,3] * Γl[3,b,c] +
              gu[a,4] * Γl[4,b,c]
              for a in 1:D, b in 1:D, c in 1:D]
end

function geodesic(r::SVector{2D,T}, metric::Metric, λ::T)::SVector{2D,T} where {Metric, T}
    x = SVector{D}(r[1:D])
    u = SVector{D}(r[D+1:end])
    Γ = christoffel(metric, SVector{D}(x))
    xdot = u
    udot = @SVector T[- sum(@SMatrix T[Γ[a,x,y] * u[x] * u[y]
                                       for x in 1:D, y in 1:D])
                      for a in 1:D]
    SVector{2D}(vcat(xdot, udot))
end

function cart2polar(x::AbstractVector{T})::Array{T,1} where T<:Real
    th = acos(x[3]/norm(x)) # in [0,pi]
    ph = sign(x[2])*acos(x[1]/norm(x[1:2])) #atan(x[2],x[1]) # in [0,pi]   # in (-pi/2,pi/2)
    [th,ph]
end

function get_background(filename::String)
    background = load(filename)
    h,w = size(background)
    th_b = LinRange(0,h,h)
    ph_b = LinRange(0,w,w)
    th_b = th_b*(pi/h)
    ph_b = pi .- ph_b*(2*pi/(w)) #pi/2 .- ph_b*(pi/(w))
    back_angs = [[th_b[i],ph_b[j]] for i in 1:h, j in 1:w]
    background, back_angs
end

angle2rad(x) = x/180*pi

function get_rotation_matrix(roll::T,pitch::T,yaw::T)::AbstractArray where T<:Real

    R_euler = RotXYZ(angle2rad(roll),
                     angle2rad(pitch),
                     angle2rad(yaw))
    R_euler
end

function get_velocity_angles(v_fov::T, h_fov::T, dist::Float64,
                             pixels_h::Int64, pixels_w::Int64) where T<:Real
    v_fov = angle2rad(v_fov)
    h_fov = angle2rad(h_fov)

    ### works only for fovs < 90 degs
    z_max = abs(dist/tan(abs(pi/2-v_fov/2)))
    y_max = abs(dist/tan(abs(pi/2-h_fov/2)))

    pixels_h==1 ? y = 0 : z = LinRange(-z_max,z_max,pixels_h)
    pixels_w==1 ? z = 0 : y = LinRange(-y_max,y_max,pixels_w)

    th = @. atan(dist,z) - pi/2 #vertical
    ph = @. atan(dist,y) - pi/2 #horizontal
    th, ph
end

function get_final_velocity(normal::AbstractVector{T},
                            g::Metric, gu::Metric) where {Metric,T<:Real}
    normal = SVector{D,T}(normal./norm(normal))
    t = gu * SVector{D,T}(1, 0, 0, 0)
    t2 = t' * g * t
    n2 = normal' * g * normal
    u = (t / sqrt(-t2) + normal / sqrt(n2)) / sqrt(T(2))
    u
end

function haversine(v1::AbstractVector{T},v2::AbstractVector{T})::T where T<:Real
   th1, ph1 = v1
   th2, ph2 = v2

   th1 = th1 - pi/2
   th2 = th2 - pi/2

   d = asin(sqrt(
       ( sin((th1-th1)/2 ))^2 + cos(th1)*cos(th1)* ( sin( (ph2-ph1)/2 ) )^2 )
   )
   println(d)
   abs(d)
end

function update_image!(img::Array{RGBA{N0f8},2},
                       th::Vector{T}, ph::Vector{T},
                       g::SMatrix{4, 4, Float64, 16}, gu::SMatrix{4, 4, Float64, 16},
                       tspan::Tuple{T,T}, tol::Real,
                       pos::Vector{T}, R_euler::AbstractArray{T},
                       background::Array{RGB{N0f8},2}, back_angs::Matrix{Vector{Float64}},
                       metric::Metric
                      ) where {Metric, T<:Real}

    condition(s, λ, integrator) = (3 < norm(s[2:D]) < 4) && (abs(cart2polar(s[2:D])[1] - pi/2) < 1e-2)
    function affect!(integrator)
        terminate!(integrator)
    end
    cb = DiscreteCallback(condition, affect!)

    init_r = norm(pos[2:end])
    h,w = size(background)


    N = length(th)
    M = length(ph)
    println((N,M))
    p = Progress(N*M)

    #Threads.@threads

#         fig = Figure()
#         ax = Axis3(fig[1,1])
#         hidedecorations!(ax)  # hides ticks, grid and lables
#         hidespines!(ax)  # hide the frame
#         mesh!(ax,Sphere(Point3f(0), 1.5f0),color=:black)
#         xlims!(ax,-max(5,xyz[1][1]),max(5,xyz[1][1]))
#         ylims!(ax,-max(5,xyz[1][2]),max(5,xyz[1][2]))
#         zlims!(ax,-max(5,xyz[1][3]),max(5,xyz[1][3]))
        #println(xyz[end]," ",fin_r)
#   Threads.@threads
    @inbounds Threads.@threads for k in CartesianIndices((N,M))
        i,j = Tuple(k)
        #println(Tuple(k))
        #### Initial setting for pos = [0,10,0,0]
        #normal = -pos./norm(pos)
        normal = T[0, -1 * sign(pos[2])*1,0,0]
        fov = T[0,sin(ph[j]),sin(th[i])]
        #fov = T[0,sin(pi/12),sin(angle2rad(5))]

        ##### Rotate normal Velocities appropriately ########
        normal[2:end] = R_euler*(normal[2:end]+fov)

        u = get_final_velocity(normal,g,gu)

        prob = ODEProblem(geodesic, SVector{2D,T}(vcat(pos,u)), tspan, metric)

        sims = solve(prob, AutoTsit5(Rosenbrock23(autodiff=false)), isoutofdomain=(y,p,t)->((x->x>init_r*3)(norm(y[2:D])) && any(isnan,y)),
                    callback=cb, abstol=tol, reltol=tol, #unstable_check=(y,p,t)->any(isnan,y),
                    maxiters=1e4, verbose=false)#, dtmin =1e-12, force_dtmin=true)

#         xyz = [Tuple(sims.u[i][2:D]) for i in 1:length(sims.t)]
        xyz = [sims.u[i][2:D] for i in 1:length(sims.t)]
        fin_r = norm(xyz[end])

#         lines!(ax,xyz)

        if fin_r>init_r
            angs = cart2polar(xyz[end])
            norms = [norm(back_angs[k,l]-angs,Inf) for k in 1:h, l in 1:w]
            #norms = [haversine(back_angs[i,j],angs) for i in 1:h, j in 1:w]
            idx = argmin(norms)
            img[i,j] = background[idx]

        elseif 3 < fin_r < 4
            img[i,j] = colorant"white"
        else
            img[i,j] = colorant"black"
        end

        next!(p)
    end
    finish!(p)
#     display(fig)
end

function trace_rays(metric)
    T = Float64

#     fig = Figure()
#     ax = Axis3(fig[1,1])
#     hidedecorations!(ax)  # hides ticks, grid and lables
#     hidespines!(ax)  # hide the frame
#     mesh!(ax,Sphere(Point3f(0), 1.5f0),color=:black)
#     xlims!(ax,-max(5,pos[2]),max(5,pos[2]))
#     ylims!(ax,-max(5,pos[3]),max(5,pos[3]))
#     zlims!(ax,-max(5,pos[4]),max(5,pos[4]))


    ### Initial Conditions ###
    init_r = T(30)
    dist = T(1)
    λ0,λ1 = T(0),T(100)
    tspan = (λ0, λ1)
    pos =  T[0,-init_r,0,0]

    ####### Background Image ##########
    background, back_angs = get_background("eso0932a.jpg")#("eso0932a.jpg")#("a.jpg")

    ####### Rotations ##########
    R_euler = get_rotation_matrix(0,-5,0)
    pos[2:end] = R_euler*pos[2:end]


    ####### Velocities ########
    h,w = size(background)
    N = 5
    th, ph = get_velocity_angles(90,120,dist,
                                 h÷N,w÷N)

    g = metric(SVector{D,T}(pos))
    gu = inv(g)

    img = Array{RGBA{N0f8},2}(undef,length(th),length(ph))

#     function input_func(i::Int)
#         #x = SVector{D,T}(0,10,p[i]*2.95425,0)
#        normal = SVector{D,T}(0, -1, sin(ph[i]), sin(th[i]) )
#         t = gu * SVector{D,T}(1, 0, 0, 0)
#         t2 = t' * g * t
#         n2 = normal' * g * normal
#         u = (t / sqrt(-t2) + normal / sqrt(n2)) / sqrt(T(2))
#         SVector{2D,T}(vcat(pos,u))
#         prob = ODEProblem(geodesic, input_func(i), (λ0, λ1), metric)

#     end
    #isoutofdomain=(y,p,t)->((x->x>max_r)(norm(y[2:D]))),
#     prob = ODEProblem(geodesic, input_func(1), (λ0, λ1), metric)
#     function prob_func(prob,i,repeat)
#         ODEProblem(geodesic, input_func(i), (λ0, λ1), metric)
#     end

#     probs = EnsembleProblem(prob,
#                             prob_func=prob_func)
# #     println("yes")

    tol=1e-12
    if typeof(ph) == T
        update_image!(img,
                    [th], [ph],
                    g, gu,
                    tspan, tol,
                    pos, R_euler,
                    background ,back_angs,
                    metric
                    )
    else
        update_image!(img,
                    th, ph,
                    g, gu,
                    tspan, tol,
                    pos, R_euler,
                    background ,back_angs,
                    metric
                    )
    end
    #display(fig)
    imshow(img)
    save("win.png",img)
end


end
