# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearDiffusionEquation3D(diffusivity)

The linear diffusion equation (or heat equation) in three space dimensions with constant `diffusivity` ``\kappa``:
```math
\partial_t u = \partial_1 \left( \kappa \partial_1 u \right)
             + \partial_2 \left( \kappa \partial_2 u \right)
             + \partial_3 \left( \kappa \partial_3 u \right).
```
Unlike [`LaplaceDiffusion3D`](@ref), which represents the parabolic part of a
hyperbolic-parabolic equation, `LinearDiffusionEquation3D` represents a purely parabolic
equation without any hyperbolic part.
"""
struct LinearDiffusionEquation3D{RealT <: Real} <: AbstractLaplaceDiffusion{3, 1}
    diffusivity::RealT
end

# Together with our specialization of `Adapt.adapt_structure`,
# this allows to move semidiscretizations and their components including
# the equations to GPUs and adapt the floating point type, e.g.,
# to `Float32` to improve performance on GPUs.
function Base.similar(equations::LinearDiffusionEquation3D,
                      ::Type{NewRealT}) where {NewRealT}
    return LinearDiffusionEquation3D(convert(NewRealT, equations.diffusivity))
end

varnames(::typeof(cons2cons), ::LinearDiffusionEquation3D) = ("scalar",)
varnames(::typeof(cons2prim), ::LinearDiffusionEquation3D) = ("scalar",)
varnames(::typeof(cons2entropy), ::LinearDiffusionEquation3D) = ("scalar",)

@inline cons2prim(u, equations::LinearDiffusionEquation3D) = u
@inline cons2entropy(u, equations::LinearDiffusionEquation3D) = u

@inline entropy(u::Real, ::LinearDiffusionEquation3D) = 0.5f0 * u^2
@inline entropy(u, equations::LinearDiffusionEquation3D) = entropy(u[1], equations)

@inline function flux(u, gradients, orientation::Integer,
                      equations::LinearDiffusionEquation3D)
    dudx, dudy, dudz = gradients
    if orientation == 1
        return SVector(equations.diffusivity * dudx)
    elseif orientation == 2
        return SVector(equations.diffusivity * dudy)
    else # if orientation == 3
        return SVector(equations.diffusivity * dudz)
    end
end
end # @muladd
