"""
    gradient_variable_transformation(equations::AbstractEquationsParabolic)

Return the mapping from conservative variables to the variables in which parabolic
gradients are computed. Defaults to [`cons2cons`](@ref), may be specialized to [`cons2prim`](@ref) 
or [`cons2entropy`](@ref) depending on the equation and type of `gradient_variables`. 
"""
gradient_variable_transformation(::AbstractEquationsParabolic) = cons2cons

# By default, the gradients are taken with respect to the conservative variables.
# this is reflected by the type parameter `GradientVariablesConservative` in the abstract
# type `AbstractEquationsParabolic{NDIMS, NVARS, GradientVariablesConservative}`.
struct GradientVariablesConservative end

"""
    have_space_time_dependent_flux(equations::AbstractEquationsParabolic)

Trait function determining whether `equations` use the parabolic flux signature
`flux(u, gradients, orientation, x, t, equations)`. The default is `False()`, selecting
the legacy space- and time-independent flux signature.

Custom equations using coordinates or time in their parabolic flux must return `True()`.
"""
@inline have_space_time_dependent_flux(::AbstractEquationsParabolic) = False()

"""
    flux(u, gradients, orientation, x, t, equations::AbstractEquationsParabolic)

Calculate a parabolic flux at coordinates `x` and time `t`.

The default implementation forwards to the time-independent method
`flux(u, gradients, orientation, equations)` to preserve compatibility with existing
parabolic equation implementations.
"""
@inline function flux(u, gradients, orientation::Integer, x, t,
                      equations::AbstractEquationsParabolic)
    return flux(u, gradients, orientation, equations)
end

"""
    max_diffusivity(u, x, t, equations::AbstractEquationsParabolic)

Return a local upper bound for the diffusivity at state `u`, coordinates `x`, and time `t`.

The default implementation forwards to `max_diffusivity(u, equations)`.
"""
@inline function max_diffusivity(u, x, t, equations::AbstractEquationsParabolic)
    return max_diffusivity(u, equations)
end

include("diffusivity_coefficients.jl")

include("laplace_diffusion.jl")

include("laplace_diffusion_entropy_variables.jl")

include("linear_diffusion_equation.jl")

include("compressible_navier_stokes.jl")
