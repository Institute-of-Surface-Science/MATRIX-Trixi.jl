# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    ZeroFluxEquations1D(nvariables; RealT = Float64)

A one-dimensional system with `nvariables` conservative variables and identically zero
hyperbolic flux,
```math
\partial_t u = 0.
```
This can be used as the hyperbolic carrier in a [`SemidiscretizationHyperbolicParabolic`](@ref)
when all dynamics belong to the parabolic or implicit part of a split problem.
`RealT` controls the scalar type used to represent the zero characteristic speed.
"""
struct ZeroFluxEquations1D{NVARS, RealT <: Real} <: AbstractEquations{1, NVARS} end

function ZeroFluxEquations1D(nvariables::Integer; RealT::Type{<:Real} = Float64)
    nvariables_int = Int(nvariables)
    if nvariables_int < 1
        throw(ArgumentError("`nvariables` must be positive."))
    end

    return ZeroFluxEquations1D{nvariables_int, RealT}()
end

function Base.similar(::ZeroFluxEquations1D{NVARS},
                      ::Type{NewRealT}) where {NVARS, NewRealT}
    return ZeroFluxEquations1D{NVARS, NewRealT}()
end

function varnames(::typeof(cons2cons), ::ZeroFluxEquations1D{NVARS}) where {NVARS}
    return ntuple(variable -> "scalar_$(variable)", Val(NVARS))
end

function varnames(::typeof(cons2prim), equations::ZeroFluxEquations1D)
    return varnames(cons2cons, equations)
end

function varnames(::typeof(cons2entropy), equations::ZeroFluxEquations1D)
    return varnames(cons2cons, equations)
end

@inline flux(u, orientation::Integer, ::ZeroFluxEquations1D) = zero(u)

@inline have_constant_speed(::ZeroFluxEquations1D) = True()

@inline function max_abs_speeds(::ZeroFluxEquations1D{NVARS, RealT}) where {NVARS,
                                                                            RealT}
    return SVector(zero(RealT))
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     ::ZeroFluxEquations1D)
    return zero(promote_type(eltype(u_ll), eltype(u_rr)))
end

@inline cons2prim(u, ::ZeroFluxEquations1D) = u
@inline cons2entropy(u, ::ZeroFluxEquations1D) = u
@inline entropy(u, ::ZeroFluxEquations1D) = 0.5f0 * sum(abs2, u)
end # @muladd
