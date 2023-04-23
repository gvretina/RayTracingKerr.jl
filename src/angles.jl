module angles

using LinearAlgebra
using StaticArrays
using Rotations

function cart2polar(x::AbstractVector{T})::Array{T,1} where {T<:Real}
    th = acos(x[3] / norm(x)) # in [0,pi]
    ph = sign(x[2]) * acos(x[1] / norm(x[1:2])) #atan(x[2],x[1]) # in [0,pi]   # in (-pi/2,pi/2)
    [th, ph]
end

function get_rotation_matrix(roll::T, pitch::T, yaw::T)::AbstractArray where {T<:Real}

    R_euler = RotXYZ(deg2rad(roll),
        deg2rad(pitch),
        deg2rad(yaw))
    R_euler
end

function get_velocity_angles(v_fov::T, h_fov::T, dist::Float64,
    pixels_h::Int64, pixels_w::Int64) where {T<:Real}
    v_fov = deg2rad(v_fov)
    h_fov = deg2rad(h_fov)

    ### works only for fovs < 90 degs
    z_max = abs(dist / tan(abs(pi / 2 - v_fov / 2)))
    y_max = abs(dist / tan(abs(pi / 2 - h_fov / 2)))

    pixels_h == 1 ? y = 0 : z = LinRange(-z_max, z_max, pixels_h)
    pixels_w == 1 ? z = 0 : y = LinRange(-y_max, y_max, pixels_w)

    th = @. atan(dist, z) - pi / 2 #vertical
    ph = @. atan(dist, y) - pi / 2 #horizontal
    th, ph
end

function get_final_velocity(normal::AbstractVector{T},
    g::Metric, gu::Metric) where {Metric,T<:Real}
    normal = SVector{D,T}(normal ./ norm(normal))
    t = gu * SVector{D,T}(1, 0, 0, 0)
    t2 = t' * g * t
    n2 = normal' * g * normal
    u = (t / sqrt(-t2) + normal / sqrt(n2)) / sqrt(T(2))
    u
end

function haversine(v1::AbstractVector{T}, v2::AbstractVector{T})::T where {T<:Real}
    th1, ph1 = v1
    th2, ph2 = v2

    th1 = th1 - pi / 2
    th2 = th2 - pi / 2

    d = asin(sqrt(
        (sin((th1 - th1) / 2))^2 + cos(th1) * cos(th1) * (sin((ph2 - ph1) / 2))^2)
    )
    println(d)
    abs(d)
end

end
