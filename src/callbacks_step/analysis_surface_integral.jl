# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains analysis computations that are performed on the surface, 
# such as aerodynamic coefficients.

"""
    AnalysisSurfaceIntegral{Variable, NBoundaries}(boundary_symbols::NTuple{NBoundaries, Symbol},
                                                   variable)

This struct is used to compute the surface integral of a quantity of interest `variable` alongside
the boundary/boundaries associated with particular names given in `boundary_symbols`.
For instance, this can be used to compute the lift [`LiftCoefficientPressure2D`](@ref) or
drag coefficient [`DragCoefficientPressure2D`](@ref) of e.g. an 2D airfoil with the boundary
names `:AirfoilTop`, `:AirfoilBottom` which would be supplied as 
`boundary_symbols = (:AirfoilTop, :AirfoilBottom)`.
A single boundary name can also be supplied, e.g. `boundary_symbols = (:AirfoilTop,)`.

- `boundary_symbols::NTuple{NBoundaries, Symbol}`: Name(s) of the boundary/boundaries
  where the quantity of interest is computed
- `variable::Variable`: Quantity of interest, like lift or drag
"""
struct AnalysisSurfaceIntegral{Variable, NBoundaries}
    variable::Variable # Quantity of interest, like lift or drag
    boundary_symbols::NTuple{NBoundaries, Symbol} # Name(s) of the boundary/boundaries

    function AnalysisSurfaceIntegral(boundary_symbols::NTuple{NBoundaries, Symbol},
                                     variable) where {NBoundaries}
        return new{typeof(variable), NBoundaries}(variable, boundary_symbols)
    end
end

@doc raw"""
    NormalParabolicFlux(component::Integer = 1;
                        factor = 1,
                        name::Symbol = :normal_parabolic_flux)

Calculate the instantaneous outward normal parabolic flux of `component` through the
boundaries selected by [`AnalysisSurfaceIntegral`](@ref). If several boundaries are selected,
their contributions are summed.

The reported value is based on the final numerical boundary flux used by the parabolic
divergence discretization. For equations written as
```math
\partial_t u = \nabla \cdot (D \nabla u),
```
the conventional Fickian flux is obtained with `factor = -1`. The current implementation
supports purely parabolic one-dimensional [`TreeMesh`](@ref) and two-dimensional
[`P4estMesh`](@ref) semidiscretizations.

- `component::Integer`: Index of the parabolic flux component
- `factor`: Multiplicative factor applied to the outward normal flux
- `name::Symbol`: Name used in analysis output
"""
struct NormalParabolicFlux{Component, Factor}
    factor::Factor
    name::Symbol
end

function NormalParabolicFlux(component::Integer = 1;
                             factor = 1,
                             name::Symbol = :normal_parabolic_flux)
    component_int = Int(component)
    component_int > 0 ||
        throw(ArgumentError("component must be a positive integer, got $component"))
    return NormalParabolicFlux{component_int, typeof(factor)}(factor, name)
end

# This returns the boundary indices of a given iterable datastructure of boundary symbols.
function get_boundary_indices(boundary_symbols, boundary_symbol_indices)
    indices = Int[]
    for name in boundary_symbols
        append!(indices, boundary_symbol_indices[name])
    end
    sort!(indices) # Try to achieve some data locality by sorting

    return indices
end

struct ForceState{RealT <: Real, NDIMS}
    psi::NTuple{NDIMS, RealT} # Unit vector normal or parallel to freestream
    rho_inf::RealT
    u_inf::RealT
    l_inf::RealT
end

# Abstract base type used for dispatch of `analyze` for quantities
# requiring gradients of the velocity field.
abstract type VariableParabolic end

struct LiftCoefficientPressure{RealT <: Real, NDIMS}
    force_state::ForceState{RealT, NDIMS}
end

struct DragCoefficientPressure{RealT <: Real, NDIMS}
    force_state::ForceState{RealT, NDIMS}
end

struct LiftCoefficientShearStress{RealT <: Real, NDIMS} <: VariableParabolic
    force_state::ForceState{RealT, NDIMS}
end

struct DragCoefficientShearStress{RealT <: Real, NDIMS} <: VariableParabolic
    force_state::ForceState{RealT, NDIMS}
end

function (lift_coefficient::LiftCoefficientPressure)(u, normal_direction, x, t,
                                                     equations)
    p = pressure(u, equations)
    @unpack psi, rho_inf, u_inf, l_inf = lift_coefficient.force_state
    # Normalize as `normal_direction` is not necessarily a unit vector
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5f0 * rho_inf * u_inf^2 * l_inf)
end

function (drag_coefficient::DragCoefficientPressure)(u, normal_direction, x, t,
                                                     equations)
    p = pressure(u, equations)
    @unpack psi, rho_inf, u_inf, l_inf = drag_coefficient.force_state
    # Normalize as `normal_direction` is not necessarily a unit vector
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5f0 * rho_inf * u_inf^2 * l_inf)
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:LiftCoefficientPressure{<:Any,
                                                                               <:Any}})
    return "CL_p"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:LiftCoefficientPressure{<:Any,
                                                                             <:Any}})
    return "CL_p"
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:DragCoefficientPressure{<:Any,
                                                                               <:Any}})
    return "CD_p"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:DragCoefficientPressure{<:Any,
                                                                             <:Any}})
    return "CD_p"
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:LiftCoefficientShearStress{<:Any,
                                                                                  <:Any}})
    return "CL_f"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:LiftCoefficientShearStress{<:Any,
                                                                                <:Any}})
    return "CL_f"
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:DragCoefficientShearStress{<:Any,
                                                                                  <:Any}})
    return "CD_f"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:DragCoefficientShearStress{<:Any,
                                                                                <:Any}})
    return "CD_f"
end

function pretty_form_ascii(surface_variable::AnalysisSurfaceIntegral{Variable}) where {
                                                                                       Variable <:
                                                                                       NormalParabolicFlux}
    return String(surface_variable.variable.name)
end

function pretty_form_utf(surface_variable::AnalysisSurfaceIntegral{Variable}) where {
                                                                                     Variable <:
                                                                                     NormalParabolicFlux}
    return String(surface_variable.variable.name)
end

include("analysis_surface_integral_1d.jl")
include("analysis_surface_integral_2d.jl")
include("analysis_surface_integral_3d.jl")
end # muladd
