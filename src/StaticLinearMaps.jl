module StaticLinearMaps
using StaticArrays

import Base: +, -, *, \, /, ==, transpose
import StaticArrays: Size
export SAdjoint, SScaling, SComposition, domain, codomain, MProjection

if VERSION >= v"0.7.0-DEV.1415"
    const adjoint = Base.adjoint
else
    const adjoint = Base.ctranspose
end

abstract type SLinearMap{M,N} end

Size(::SLinearMap{M,N}) where {M,N} = (M,N)
Size(::Type{<:SLinearMap{M,N}}) where {M,N} = (M,N)

struct SScaling{N,T} <: SLinearMap{N,N}
    λ::T
end

*(a::SScaling, x::StaticVector) = a.λ*x
#*(x::StaticVector, a::SScaling) = x*a.λ
*(x::RowVector{T,S}, a::SScaling) where {T,S<:StaticVector} = x*a.λ

struct SAdjoint{N,M,T<:SLinearMap{M,N}} <: SLinearMap{N,M}
    f::T
end

adjoint(t::T) where {T<:SLinearMap{M,N}} where {M,N} = SAdjoint{N,M,T}(t)

#*(a::SAdjoint, x) = (x'*a.f)' # this fallback is too likely to produce stackoverflow errors
*(x, a::SAdjoint) = (a.f*x')'
*(x, a::T) where {T<:SLinearMap} = (a'*x')'

struct SComposition{M,N,P,S<:SLinearMap{M,N},T<:SLinearMap{N,P}} <: SLinearMap{M,P}
    f::S
    g::T
end

*(f::S, g::T) where {S<:SLinearMap{M,N},T<:SLinearMap{N,P}} where {M,N,P} = SComposition{M,N,P,S,T}(f,g)

*(c::SComposition, x) = c.f*(c.g*x)
*(y, c::SComposition) = (y*c.f)*c.g



struct SProjection{M,N,S<:SVector{M,Int}} <: SLinearMap{M,N}
    p::S
end

@generated function *(P::SProjection{M,N}, x::StaticVector{N,T}) where {M,N,T}
    expr = [:(x[P.p[$i]]) for i in 1:M]
    :(SVector{M,T}($(expr...)))
end

struct SInjection{M,N,S<:SVector{M,Int}} <: SLinearMap{M,N}
    p::S
end

@generated function *(P::SInjection{M,N}, x::StaticVector{N,T}) where {M,N,T}
    expr = [:(P.p[$i] > 0 ? x[P.p[$i]]' : zero(T)') for i in 1:M]
    :(SVector{M,T}($(expr...)))
end

function *(PT::SAdjoint{N,M,Q}, x::StaticVector{M,T}) where {N,M,T,Q<:SProjection}
    P = PT.f
    out = zero(MVector{N,T})
    for i in 1:length(P.p)
        out[P.p[i]] = x[i]
    end
    out
end

struct MProjection{M,N,S} <: SLinearMap{M,N}
    d::S
end

function *(P::MProjection{M,N}, x::StaticVector{N,T}) where {M,N,T}
    out = zero(MVector{M, T})
    for i in 1:length(P.d)
        out[P.d[i][2]] = x[P.d[i][1]]
    end
    out
end
function *(PT::SAdjoint{N,M,Q}, x::StaticVector{M,T}) where {N,M,T,Q<:MProjection}
    P = PT.f
    out = zero(MVector{N,T})
    for i in 1:length(P.d)
        out[P.d[i][1]] = x[P.d[i][2]]
    end
    out
end

end # module
