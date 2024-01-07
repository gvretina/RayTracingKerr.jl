using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using FLoops
using DifferentialEquations
using LinearAlgebra
using StaticArrays
using ForwardDiff
using OrdinaryDiffEq
using IterTools
using ProgressMeter
using Colors
using Images#,ImageView
using Rotations
using LazyArrays

const D = 4
const T = Float64
const pos_mask = [true, true, true, true,false, false, false, false]
const vel_mask = .~pos_mask
const cart_mask = [false, true, true, true,false, false, false, false]
function kerr_schild(xx::SVector{D,T})::SMatrix{D,D,T} where {T}
    M = 1

    t,x,y,z = xx
    @fastmath begin
    # <https://en.wikipedia.org/wiki/Kerr_metric>
    η = @MMatrix T[a==b ? (a==1 ? -1 : 1) : 0 for a in 1:D, b in 1:D]
    a = T(0.9)                       # T(0.8)

    ρ = sqrt(x^2 + y^2 + z^2)
    if ρ^2 - a^2<0
        r = NaN
    else
        r = sqrt( (ρ^2 - a^2)/2 + sqrt(a^2*z^2 + ((ρ^2 - a^2)/2)^2) )
    end
    f = 2*M*r^3 / (r^4 + a^2*z^2)
    k = SVector{D,T}(1,
                     (r*x + a*y) / (r^2 + a^2),
                     (r*y - a*x) / (r^2 + a^2),
                     z / r)

    g = @SMatrix T[η[a,b] + f * k[a] * k[b] for a in 1:D, b in 1:D]
    end
    g
end

function christoffel(metric::Metric, x::SVector{D,T}) where {Metric, T}
    g = metric(x)
    dg = reshape(ForwardDiff.jacobian(metric,x),(D,D,D))
    @fastmath begin

    gu = inv(g)
    Γl = @SArray T[(dg[a,b,c] + dg[a,c,b] - dg[b,c,a]) / 2
                   for a in 1:D, b in 1:D, c in 1:D]
    @SArray T[gu[a,1] * Γl[1,b,c] +
              gu[a,2] * Γl[2,b,c] +
              gu[a,3] * Γl[3,b,c] +
              gu[a,4] * Γl[4,b,c]
              for a in 1:D, b in 1:D, c in 1:D]
    end
end

function geodesic!(du::AbstractArray, r::AbstractArray, metric::Metric, λ::T) where {Metric, T}
    x = SVector{D,T}(r[pos_mask])
    u = SVector{D,T}(r[vel_mask])
    Γ = christoffel(metric, x)

    @fastmath begin
        @inbounds for idx in 1:D
            du[idx] = r[idx+D]
            du[D+idx] = -sum(@MMatrix T[Γ[idx,b,c] * u[b] * u[c]
                                            for b in 1:D, c in 1:D])
        end
    end
end

function geodesic!(du::MVector{2D,T}, r::MVector{2D,T}, metric::Metric, λ::T) where {Metric, T}
    x = SVector{D,T}(r[pos_mask])
    u = SVector{D,T}(r[vel_mask])
    Γ = christoffel(metric, x)

    @fastmath begin
        @inbounds for idx in 1:D
            du[idx] = r[idx+D]
            du[D+idx] = -sum(@SMatrix T[Γ[idx,b,c] * u[b] * u[c]
                                            for b in 1:D, c in 1:D])
        end
    end
end

function cart2polar(x::AbstractVector{T})::SVector{2,T} where T<:Real
    th = acos(x[3]/norm(x)) # in [0,pi]
    r = SVector{2}( @views x[1:2])
    if x[2]>0
        ph = acos(x[1]/norm(r)) #atan(x[2],x[1]) # in [0,pi]   # in (-pi/2,pi/2)
    else
        ph = -1*acos(x[1]/norm(r)) #atan(x[2],x[1]) # in [0,pi]   # in (-pi/2,pi/2)
    end
    SVector{2,T}(ApplyArray(vcat,th,ph))
end

function get_background(filename::String)
    background = Images.load(filename)
    h,w = size(background)
    th_b = LinRange(0,h,h)
    ph_b = LinRange(0,w,w)
    th_b = th_b*(pi/h)
    ph_b = pi .- ph_b*(2*pi/(w)) #pi/2 .- ph_b*(pi/(w))
    back_angs = Matrix{Vector{Float64}}(undef, h, w)
    for j in 1:w
        for i in 1:h
            back_angs[i,j] = [th_b[i],ph_b[j]]
        end
    end
    back_angs = [[th_b[i],ph_b[j]] for j in 1:w, i in 1:h]
    if filename == "a.jpg"
        background = Matrix(imrotate(background,-pi/2)')
    end
    background, back_angs
end

function get_rotation_matrix(roll::Number,pitch::Number,yaw::Number)::AbstractArray

    R_euler = RotXYZ(deg2rad(roll),
                     deg2rad(pitch),
                     deg2rad(yaw))

    SMatrix{D,D,Float64,16}(ApplyArray(vcat,[0,0,0,0.]',ApplyArray(hcat,[0.,0,0],R_euler)))
end

function get_velocity_angles(v_fov::T, h_fov::T, dist::Float64,
                             pixels_h::Int64, pixels_w::Int64)::Vector{Vector{Float64}} where T<:Real

    ### works only for fovs < 90 degs
    z_max = abs(dist/tan(abs(pi/2 - deg2rad(v_fov)/2)))
    y_max = abs(dist/tan(abs(pi/2 - deg2rad(h_fov)/2)))

    pixels_h==1 ? th = [0.] : th = atan.(dist, LinRange(-z_max,z_max,pixels_h)) .- pi/2
    pixels_w==1 ? ph = [0.] : ph = atan.(dist, LinRange(-y_max,y_max,pixels_w)) .- pi/2

    [th, ph]
end

function get_final_velocity(normal::MVector{D,T},
                            g::Metric, gu::Metric)::MVector{D,T} where {Metric,T<:Real}
    @fastmath begin
    normal = MVector{D,T}(normal./norm(normal))
    t = gu * MVector{D,T}(1, 0, 0, 0)
    t2 = t' * g * t
    n2 = normal' * g * normal
    normal = (t / sqrt(-t2) + normal / sqrt(n2)) / sqrt(T(2))
    end
    normal
end

function get_background_pixel_idx(back_angs::Matrix{Vector{T}},
                      angs::SVector{2,T})::CartesianIndex{2} where T

    norm_mat = argmin([abs2.(SVector{2,T}(back_angs[m,n]-angs)) for m in 1:h, n in 1:w])
    norm_mat
end

const background, back_angs = get_background("../scene.jpg")#eso0932a
const h, w = size(background)
const init_r = Float64(15)
const lower_r = 3.0
const higher_r = 6.0
const bh_hor = 1.8

condition(s, λ, integrator) = (
    ((lower_r < norm(SVector{3,T}(s[cart_mask])) < higher_r)
        && (abs(cart2polar(SVector{3,T}(integrator.u[cart_mask]))[1] - pi/2) < 1e-2)) ### hits accretion disk
        || (norm(SVector{3,T}(@view integrator.u[cart_mask])) > 1.1*init_r) ### goes to infty
        || ((norm(SVector{3,T}(@view integrator.u[cart_mask])) < bh_hor) ### hits BH horizon r ~ 1.435889
           && ( abs(dot(integrator.u[vel_mask],integrator.p(SVector{D,T}(integrator.u[pos_mask])),integrator.u[pos_mask])) > 1e-3 )
            ) ### stops being null
        )

function affect!(integrator)
    terminate!(integrator)
end
const cb = DiscreteCallback(condition, affect!)

function update_image_2!(img::AbstractArray,
                        i::Int64,j::Int64,
                         pixels_h::Int64, pixels_w::Int64,
                         tspan::Tuple{T,T}, tol::Real,
                         input::MVector{2D, T},
                         metric::Metric) where {Metric, T<:Real}

    function constraints(resid,u,p,t)
        g = p(@SVector [u[i] for i in 1:D])
        v = u[vel_mask]
        resid[1] = dot(v,g,v)
        resid[2:2D] .= 0
    end

    cbb = ManifoldProjection(constraints,nlopts=Dict(:method => :anderson,:m=>0))

    prob = ODEProblem(geodesic!, input, tspan, metric)

    (@isdefined cbb) ? cbb=CallbackSet(cb,cbb) : cbb=cb
    sims = solve(prob, AutoVern9(Rodas5()), callback=cbb, #callback=cb, abstol=tol, reltol=tol,
                abstol=tol, reltol=tol, save_everystep=false, save_start=false,# dt=1e-2,
                maxiters=1e5, verbose=false)#,dtmin=1e-3,force_dtmin=true)#, save_idxs = [2,3,4])

    fin_r = norm(SVector{3,T}(sims.u[end][cart_mask]))

    if sims.retcode == ReturnCode.Success || sims.retcode == ReturnCode.MaxIters
        if fin_r > init_r
            angs = cart2polar(SVector{3,T}(sims.u[end][cart_mask]))
            idx = get_background_pixel_idx(back_angs,angs)
            img[i,j] = background[idx]
        else
            img[i,j] = RGB{N0f8}(0.0, 0.0, 0.0)
        end
    elseif sims.retcode == ReturnCode.Terminated
        if fin_r > init_r
            angs = cart2polar(SVector{3,T}(sims.u[end][cart_mask]))
            idx = get_background_pixel_idx(back_angs,angs)
            img[i,j] = background[idx]
        elseif ((lower_r < fin_r < higher_r &&
                abs(cart2polar(SVector{3,T}(sims.u[end][cart_mask]))[1] - pi/2) < 1e-2) ||
                bh_hor < fin_r < lower_r) 
            img[i,j] = RGB{N0f8}(1.0, 1.0, 1.0)
        else
            img[i,j] = RGB{N0f8}(0.0, 0.0, 0.0)
        end
    elseif sims.retcode == ReturnCode.Unstable
        img[i,j] = RGB{N0f8}(0.0, 0.0, 0.0)
    else
        img[i,j] = colorant"purple"
    end

end




function trace_rays(metric::Metric) where Metric
    trace_rays(metric,10)
end

function trace_rays(metric::Metric, sc::Int64) where Metric

    ### Initial Conditions ###
    dist = T(1)
    tspan = (T(0),T(100))

    ####### Velocities ######## TODO make matrix with all combinations and feed that to updater
    #h,w = size(background)
    scaler = h>sc ? h÷sc : -1
    if scaler != -1
        N,M = h÷scaler,w÷scaler
    else
        N,M = sc,sc
    end
    vfov = 60
    hfov = (vfov*M)÷N
    println(N," ",M)
    th, ph = get_velocity_angles(vfov,hfov,dist,
                                 N,M)
    ####### Rotations ##########
    R_euler = get_rotation_matrix(0,rad2deg(pi/18),0)
    pos = R_euler * @SVector  [0,-init_r,0,0]
    g = metric(SVector{D,T}(pos))
    gu = inv(g)
    inputs = zeros(MVector{2D,T}, N, M)

    @fastmath begin
        @inbounds for j in eachindex(ph), i in eachindex(th)
            inputs[i,j] = ApplyArray(vcat,pos,
                                      get_final_velocity(
                                          R_euler * MVector{D,T}(0,1, sin(ph[j]), sin(th[i])),
                                          g,gu)
                                      )
        end
    end


    img =Array{eltype(background),2}(undef,N,M)

    tol=1e-14#eps(T)^(3/4)

    p = Progress(N*M, showspeed=true)
    if Threads.nthreads() == 1
        ex = SequentialEx()
    else
        ex = ThreadedEx()
    end
    @floop ex for j in 1:M, i in 1:N
#     for i in 1:N, j in 1:M
        update_image_2!(img,
                        i,j,
                        N, M,
                        tspan, tol,
                        inputs[i,j],
                        metric)
        next!(p)
    end
    finish!(p)

    #imshow(img)
    save("render_win_$(N)_x_$(M).jpg",img)
end
println("Starting...")
trace_rays(kerr_schild,1000)
