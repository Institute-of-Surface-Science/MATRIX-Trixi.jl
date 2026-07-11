# Linear scalar diffusion for use in linear scalar advection-diffusion problems
abstract type AbstractLaplaceDiffusion{NDIMS, NVARS} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariablesConservative} end

"""
    have_constant_diffusivity(::AbstractLaplaceDiffusion)

# Returns
- `True()`

Used in parabolic cfl condition computation (see [`StepsizeCallback`](@ref)) to indicate that the
diffusivity is constant in space and that [`max_diffusivity`](@ref) needs **not** to be re-computed
at every node in every element.

Also employed in [`linear_structure`](@ref) and [`linear_structure_parabolic`](@ref) to check
if the diffusion term is linear in the variables/constant.
"""
@inline have_constant_diffusivity(::AbstractLaplaceDiffusion) = True()

"""
    max_diffusivity(equations_parabolic::AbstractLaplaceDiffusion)

# Returns
- `equations_parabolic.diffusivity`

Returns isotropic diffusion coefficient for use in parabolic cfl condition computation,
see [`StepsizeCallback`](@ref).
"""
@inline function max_diffusivity(equations_parabolic::AbstractLaplaceDiffusion)
    return equations_parabolic.diffusivity
end

@inline function penalty(u_outer, u_inner,
                         equations_parabolic::AbstractLaplaceDiffusion,
                         ::ParabolicFormulationLocalDG{Nothing})
    return zero(u_inner)
end

@inline function penalty(u_outer, u_inner,
                         equations_parabolic::AbstractLaplaceDiffusion,
                         dg::ParabolicFormulationLocalDG)
    return dg.penalty_parameter .* (u_outer - u_inner) .*
           equations_parabolic.diffusivity
end

@inline function penalty(u_outer, u_inner, inv_h,
                         equations_parabolic::AbstractLaplaceDiffusion,
                         dg::ParabolicFormulationLocalDG)
    return inv_h .* penalty(u_outer, u_inner, equations_parabolic, dg)
end

@inline function penalty_diffusivity(u_outer, u_inner, x, t,
                                     equations_parabolic::AbstractLaplaceDiffusion)
    return penalty_diffusivity(u_outer, u_inner, x, t,
                               have_constant_diffusivity(equations_parabolic),
                               equations_parabolic)
end

@inline function penalty_diffusivity(u_outer, u_inner, x, t, ::True,
                                     equations_parabolic::AbstractLaplaceDiffusion)
    return equations_parabolic.diffusivity
end

@inline function penalty_diffusivity(u_outer, u_inner, x, t, ::False,
                                     equations_parabolic::AbstractLaplaceDiffusion)
    diffusivity_outer = max_diffusivity(u_outer, x, t, equations_parabolic)
    diffusivity_inner = max_diffusivity(u_inner, x, t, equations_parabolic)
    return max.(diffusivity_outer, diffusivity_inner)
end

@inline function penalty(u_outer, u_inner, inv_h, x, t,
                         equations_parabolic::AbstractLaplaceDiffusion,
                         ::ParabolicFormulationLocalDG{Nothing})
    return zero(u_inner)
end

@inline function penalty(u_outer, u_inner, inv_h, x, t,
                         equations_parabolic::AbstractLaplaceDiffusion,
                         dg::ParabolicFormulationLocalDG)
    diffusivity = penalty_diffusivity(u_outer, u_inner, x, t,
                                      equations_parabolic)
    return inv_h .* dg.penalty_parameter .* (u_outer - u_inner) .* diffusivity
end

include("laplace_diffusion_1d.jl")
include("laplace_diffusion_2d.jl")
include("laplace_diffusion_3d.jl")
include("laplace_diffusion_componentwise.jl")
