# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearDiffusionEquation2D(diffusivity)

The linear diffusion equation (or heat equation) in two space dimensions with scalar,
isotropic `diffusivity` ``\kappa``:
```math
\partial_t u = \partial_1 \left( \kappa \partial_1 u \right)
             + \partial_2 \left( \kappa \partial_2 u \right).
```
`diffusivity` may be a constant real value or a [`SpatiallyVaryingDiffusivity`](@ref).
Unlike [`LaplaceDiffusion2D`](@ref), which represents the parabolic part of a 
hyperbolic-parabolic equation, `LinearDiffusionEquation2D` represents a purely parabolic
equation without any hyperbolic part.
"""
struct LinearDiffusionEquation2D{D <: AbstractDiffusivityCoefficient} <:
       AbstractLaplaceDiffusion{2, 1}
    diffusivity::D
end

function LinearDiffusionEquation2D(diffusivity::Real)
    return LinearDiffusionEquation2D(ConstantDiffusivity(diffusivity))
end

# Together with our specialization of `Adapt.adapt_structure`,
# this allows to move semidiscretizations and their components including
# the equations to GPUs and adapt the floating point type, e.g.,
# to `Float32` to improve performance on GPUs.
function Base.similar(equations::LinearDiffusionEquation2D,
                      ::Type{NewRealT}) where {NewRealT}
    return LinearDiffusionEquation2D(similar(equations.diffusivity, NewRealT))
end

@inline function have_constant_diffusivity(equations::LinearDiffusionEquation2D)
    return have_constant_diffusivity(equations.diffusivity)
end

@inline function have_space_time_dependent_flux(equations::LinearDiffusionEquation2D)
    return have_space_time_dependent_flux(equations.diffusivity)
end

@inline function max_diffusivity(u, x, t,
                                 equations::LinearDiffusionEquation2D)
    return diffusivity_upper_bound(equations.diffusivity)
end

varnames(::typeof(cons2cons), ::LinearDiffusionEquation2D) = ("scalar",)
varnames(::typeof(cons2prim), ::LinearDiffusionEquation2D) = ("scalar",)
varnames(::typeof(cons2entropy), ::LinearDiffusionEquation2D) = ("scalar",)

@inline cons2prim(u, equations::LinearDiffusionEquation2D) = u
@inline cons2entropy(u, equations::LinearDiffusionEquation2D) = u

@inline entropy(u::Real, ::LinearDiffusionEquation2D) = 0.5f0 * u^2
@inline entropy(u, equations::LinearDiffusionEquation2D) = entropy(u[1], equations)

@inline function flux(u, gradients, orientation::Integer,
                      equations::LinearDiffusionEquation2D)
    return flux(u, gradients, orientation,
                have_space_time_dependent_flux(equations), equations)
end

@inline function flux(u, gradients, orientation::Integer, ::False,
                      equations::LinearDiffusionEquation2D)
    dudx, dudy = gradients
    diffusivity = diffusivity_value(equations.diffusivity, equations)
    if orientation == 1
        return SVector(diffusivity * dudx)
    else # if orientation == 2
        return SVector(diffusivity * dudy)
    end
end

@inline function flux(u, gradients, orientation::Integer, ::True,
                      equations::LinearDiffusionEquation2D)
    throw(ArgumentError("space- or time-dependent diffusivity requires coordinates and time"))
end

@inline function flux(u, gradients, orientation::Integer, x, t,
                      equations::LinearDiffusionEquation2D)
    diffusivity = diffusivity_value(equations.diffusivity, x, t, equations)
    return SVector(diffusivity * gradients[orientation])
end
end # @muladd
