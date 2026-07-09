@doc raw"""
    LaplaceDiffusionComponentwise{NDIMS}(diffusivity, equations)
    LaplaceDiffusionComponentwise1D(diffusivity, equations)
    LaplaceDiffusionComponentwise2D(diffusivity, equations)
    LaplaceDiffusionComponentwise3D(diffusivity, equations)

`LaplaceDiffusionComponentwise` represents component-wise scalar diffusion
``\nabla \cdot (\kappa_i \nabla u_i)`` with one constant diffusivity
``\kappa_i`` per solution component of `equations`.

Unlike [`LaplaceDiffusion1D`](@ref), [`LaplaceDiffusion2D`](@ref), and
[`LaplaceDiffusion3D`](@ref), which apply one scalar diffusivity to every
solution component, `LaplaceDiffusionComponentwise` accepts a vector of
diffusivities. Components with zero diffusivity have zero parabolic flux contribution.
"""
struct LaplaceDiffusionComponentwise{NDIMS, E, N, T} <:
       AbstractLaplaceDiffusion{NDIMS, N}
    diffusivity::T
    equations_hyperbolic::E
end

function LaplaceDiffusionComponentwise{NDIMS}(diffusivity,
                                             equations_hyperbolic) where {NDIMS}
    nvars = nvariables(equations_hyperbolic)
    if length(diffusivity) != nvars
        throw(ArgumentError("Number of diffusivities must match number of variables."))
    end

    diffusivity_static = SVector{nvars}(diffusivity)
    return LaplaceDiffusionComponentwise{NDIMS,
                                         typeof(equations_hyperbolic),
                                         nvars,
                                         typeof(diffusivity_static)}(diffusivity_static,
                                                                     equations_hyperbolic)
end

LaplaceDiffusionComponentwise1D(diffusivity, equations_hyperbolic) =
    LaplaceDiffusionComponentwise{1}(diffusivity, equations_hyperbolic)

LaplaceDiffusionComponentwise2D(diffusivity, equations_hyperbolic) =
    LaplaceDiffusionComponentwise{2}(diffusivity, equations_hyperbolic)

LaplaceDiffusionComponentwise3D(diffusivity, equations_hyperbolic) =
    LaplaceDiffusionComponentwise{3}(diffusivity, equations_hyperbolic)

# Together with our specialization of `Adapt.adapt_structure`,
# this allows to move semidiscretizations and their components including
# the equations to GPUs and adapt the floating point type, e.g.,
# to `Float32` to improve performance on GPUs.
function Base.similar(equations::LaplaceDiffusionComponentwise{NDIMS},
                      ::Type{NewRealT}) where {NDIMS, NewRealT}
    diffusivity = map(equations.diffusivity) do kappa
        kappa isa AbstractFloat ? convert(NewRealT, kappa) : kappa
    end

    return LaplaceDiffusionComponentwise{NDIMS}(diffusivity,
                                                similar(equations.equations_hyperbolic,
                                                        NewRealT))
end

function varnames(variable_mapping,
                  equations_parabolic::LaplaceDiffusionComponentwise)
    return varnames(variable_mapping, equations_parabolic.equations_hyperbolic)
end

@inline have_constant_diffusivity(::LaplaceDiffusionComponentwise) = True()

@inline function max_diffusivity(equations_parabolic::LaplaceDiffusionComponentwise)
    return maximum(abs, equations_parabolic.diffusivity)
end

@inline function flux(u, gradients, orientation::Integer,
                      equations_parabolic::LaplaceDiffusionComponentwise{1})
    dudx, = gradients
    # orientation == 1
    return equations_parabolic.diffusivity .* dudx
end

@inline function flux(u, gradients, orientation::Integer,
                      equations_parabolic::LaplaceDiffusionComponentwise{2})
    dudx, dudy = gradients
    if orientation == 1
        return equations_parabolic.diffusivity .* dudx
    else # if orientation == 2
        return equations_parabolic.diffusivity .* dudy
    end
end

@inline function flux(u, gradients, orientation::Integer,
                      equations_parabolic::LaplaceDiffusionComponentwise{3})
    dudx, dudy, dudz = gradients
    if orientation == 1
        return equations_parabolic.diffusivity .* dudx
    elseif orientation == 2
        return equations_parabolic.diffusivity .* dudy
    else # if orientation == 3
        return equations_parabolic.diffusivity .* dudz
    end
end

function penalty(u_outer, u_inner, inv_h,
                 equations_parabolic::LaplaceDiffusionComponentwise,
                 dg::ParabolicFormulationLocalDG)
    return dg.penalty_parameter * (u_outer - u_inner) .*
           equations_parabolic.diffusivity
end
