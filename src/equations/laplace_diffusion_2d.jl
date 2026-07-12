@doc raw"""
    LaplaceDiffusion2D(diffusivity, equations)

`LaplaceDiffusion2D` represents a scalar diffusion term ``\nabla \cdot (\kappa\nabla u)``
with scalar, isotropic diffusivity ``\kappa`` applied to each solution component defined by
`equations`. `diffusivity` may be a constant real value or a
[`SpatiallyVaryingDiffusivity`](@ref).
This is intended for use as the parabolic part of a hyperbolic-parabolic system, where the 
hyperbolic part is defined by `equations`. For a purely parabolic diffusion equation 
without any hyperbolic part, see [`LinearDiffusionEquation2D`](@ref).
"""
struct LaplaceDiffusion2D{E, N, T <: AbstractDiffusivityCoefficient} <:
       AbstractLaplaceDiffusion{2, N}
    diffusivity::T
    equations_hyperbolic::E
end

function LaplaceDiffusion2D(diffusivity::Real, equations_hyperbolic)
    return LaplaceDiffusion2D(ConstantDiffusivity(diffusivity), equations_hyperbolic)
end

function LaplaceDiffusion2D(diffusivity::AbstractDiffusivityCoefficient,
                            equations_hyperbolic)
    return LaplaceDiffusion2D{typeof(equations_hyperbolic),
                              nvariables(equations_hyperbolic),
                              typeof(diffusivity)}(diffusivity, equations_hyperbolic)
end

# Together with our specialization of `Adapt.adapt_structure`,
# this allows to move semidiscretizations and their components including
# the equations to GPUs and adapt the floating point type, e.g.,
# to `Float32` to improve performance on GPUs.
function Base.similar(equations::LaplaceDiffusion2D, ::Type{NewRealT}) where {NewRealT}
    return LaplaceDiffusion2D(similar(equations.diffusivity, NewRealT),
                              similar(equations.equations_hyperbolic, NewRealT))
end

@inline have_constant_diffusivity(::LaplaceDiffusion2D{<:Any, <:Any, <:SpatiallyVaryingDiffusivity}) = False()
@inline have_space_time_dependent_flux(::LaplaceDiffusion2D{<:Any, <:Any, <:SpatiallyVaryingDiffusivity}) = True()

@inline function max_diffusivity(u, x, t,
                                 equations::LaplaceDiffusion2D{<:Any, <:Any,
                                                               <:SpatiallyVaryingDiffusivity})
    return diffusivity_upper_bound(equations.diffusivity)
end

function varnames(variable_mapping, equations_parabolic::LaplaceDiffusion2D)
    return varnames(variable_mapping, equations_parabolic.equations_hyperbolic)
end

function flux(u, gradients, orientation::Integer,
              equations_parabolic::LaplaceDiffusion2D{<:Any, <:Any,
                                                      <:ConstantDiffusivity})
    dudx, dudy = gradients
    diffusivity = diffusivity_value(equations_parabolic.diffusivity,
                                    equations_parabolic)
    if orientation == 1
        return SVector(diffusivity * dudx)
    else # if orientation == 2
        return SVector(diffusivity * dudy)
    end
end

@inline function flux(u, gradients, orientation::Integer, x, t,
                      equations_parabolic::LaplaceDiffusion2D{<:Any, <:Any,
                                                              <:SpatiallyVaryingDiffusivity})
    diffusivity = diffusivity_value(equations_parabolic.diffusivity, x, t,
                                    equations_parabolic)
    return SVector(diffusivity * gradients[orientation])
end

# General Dirichlet and Neumann boundary condition functions are defined in `src/equations/laplace_diffusion_1d.jl`.
