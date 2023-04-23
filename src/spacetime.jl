module spacetime

using StaticArrays
using ForwardDiff

const D = 4

function kerr_schild(xx::SVector{D,T})::SMatrix{D,D,T} where {T}
    M = 1

    t, x, y, z = xx
    η = @SMatrix T[a == b ? (a == 1 ? -1 : 1) : 0 for a in 1:D, b in 1:D]
    a = 0.9

    ρ = nm.sqrt(x^2 + y^2 + z^2)
    r = nm.sqrt(ρ^2 - a^2) / 2 + nm.sqrt(a^2 * z^2 + ((ρ^2 - a^2) / 2)^2)
    f = 2 * M * r^3 / (r^4 + a^2 * z^2)
    k = SVector{D,T}(1,
        (r * x + a * y) / (r^2 + a^2),
        (r * y - a * x) / (r^2 + a^2),
        z / r)

    g = @SMatrix T[η[a, b] + f * k[a] * k[b] for a in 1:D, b in 1:D]
    g
end

function dmetric(g::Metric, x::SVector{D,T}) where {Metric,T}
    dg = reshape(ForwardDiff.jacobian(g, x), (D, D, D))
    g, dg
end

function christoffel(metric::Metric, x::SVector{D,T}) where {Metric,T}
    g, dg = dmetric(metric, x)
    gu = inv(g(x))
    Γl = @SArray T[(dg[a, b, c] + dg[a, c, b] - dg[b, c, a]) / 2
                   for a in 1:D, b in 1:D, c in 1:D]
    @SArray T[gu[a, 1] * Γl[1, b, c] +
              gu[a, 2] * Γl[2, b, c] +
              gu[a, 3] * Γl[3, b, c] +
              gu[a, 4] * Γl[4, b, c]
              for a in 1:D, b in 1:D, c in 1:D]
end

function geodesic(r::SVector{2D,T}, metric::Metric, λ::T)::SVector{2D,T} where {Metric,T}
    x = SVector{D}(r[1:D])
    u = SVector{D}(r[D+1:end])
    Γ = christoffel(metric, SVector{D}(x))
    xdot = u
    udot = @SVector T[-sum(@SMatrix T[Γ[a, x, y] * u[x] * u[y]
                                      for x in 1:D, y in 1:D])
                      for a in 1:D]
    SVector{2D}(vcat(xdot, udot))
end


end
