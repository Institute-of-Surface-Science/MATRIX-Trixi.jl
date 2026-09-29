"""
    AbstractDiffusivityCoefficient

Abstract type for scalar isotropic diffusivity coefficient providers.

Subtypes must implement the following interface methods:

- `have_constant_diffusivity(coefficient)`, returning `True()` or `False()`;
- `have_space_time_dependent_flux(coefficient)`, returning `True()` or `False()`;
- `diffusivity_upper_bound(coefficient)`, returning a global scalar upper bound;
- `Base.similar(coefficient, NewRealT)`, adapting stored real values to `NewRealT`.

Constant providers, for which `have_space_time_dependent_flux` returns `False()`, must
implement `diffusivity_value(coefficient, equations)`. Space- or time-dependent providers,
for which it returns `True()`, must instead implement
`diffusivity_value(coefficient, x, t, equations)`.

Equation types that can supply the local solution state call
`diffusivity_value(coefficient, u, x, t, equations)`. The default implementation forwards
to the four-argument method for backward compatibility. A provider may specialize the
five-argument method when its coefficient depends on `u`.
"""
abstract type AbstractDiffusivityCoefficient end

"""
    ConstantDiffusivity(value)

Constant scalar diffusivity coefficient.
"""
struct ConstantDiffusivity{T <: Real} <: AbstractDiffusivityCoefficient
    value::T
end

@inline have_constant_diffusivity(::ConstantDiffusivity) = True()
@inline have_space_time_dependent_flux(::ConstantDiffusivity) = False()

"""
    SpatiallyVaryingDiffusivity(value_function, upper_bound)

Scalar isotropic diffusivity evaluated as `value_function(x, t, equations)`. The supplied
`upper_bound` must be finite and positive. It is used by [`StepsizeCallback`](@ref), and
must bound every diffusivity value encountered during the simulation.

Spatially varying coefficients remain linear in the solution variables. However,
`linear_structure` is currently unavailable since Trixi uses
[`have_constant_diffusivity`](@ref) for both timestep optimization and linear-operator
construction.
"""
struct SpatiallyVaryingDiffusivity{F, T <: Real} <: AbstractDiffusivityCoefficient
    value_function::F
    upper_bound::T

    function SpatiallyVaryingDiffusivity(value_function::F,
                                         upper_bound::T) where {F, T <: Real}
        if !isfinite(upper_bound) || upper_bound <= zero(upper_bound)
            throw(ArgumentError("upper_bound must be finite and positive"))
        end

        return new{F, T}(value_function, upper_bound)
    end
end

@inline have_constant_diffusivity(::SpatiallyVaryingDiffusivity) = False()
@inline have_space_time_dependent_flux(::SpatiallyVaryingDiffusivity) = True()

@inline function have_constant_diffusivity(coefficient::AbstractDiffusivityCoefficient)
    error("Interface: Must implement have_constant_diffusivity(::$(typeof(coefficient)))")
end

@inline function have_space_time_dependent_flux(coefficient::AbstractDiffusivityCoefficient)
    error("Interface: Must implement have_space_time_dependent_flux(::$(typeof(coefficient)))")
end

@inline diffusivity_value(coefficient::ConstantDiffusivity, equations) = coefficient.value

@inline function diffusivity_value(coefficient::ConstantDiffusivity, x, t, equations)
    return coefficient.value
end

@inline function diffusivity_value(coefficient::SpatiallyVaryingDiffusivity, x, t,
                                   equations)
    return coefficient.value_function(x, t, equations)
end

@inline function diffusivity_value(coefficient::AbstractDiffusivityCoefficient, u, x, t,
                                   equations)
    return diffusivity_value(coefficient, x, t, equations)
end

@inline function diffusivity_value(coefficient::AbstractDiffusivityCoefficient, args...)
    error("Interface: Must implement diffusivity_value(::$(typeof(coefficient)), ...)")
end

@inline diffusivity_upper_bound(coefficient::ConstantDiffusivity) = coefficient.value
@inline function diffusivity_upper_bound(coefficient::SpatiallyVaryingDiffusivity)
    return coefficient.upper_bound
end

@inline function diffusivity_upper_bound(coefficient::AbstractDiffusivityCoefficient)
    error("Interface: Must implement diffusivity_upper_bound(::$(typeof(coefficient)))")
end

# Fallbacks for existing equation types that still store diffusivity values directly.
@inline diffusivity_value(diffusivity, equations) = diffusivity
@inline diffusivity_value(diffusivity, x, t, equations) = diffusivity
@inline diffusivity_value(diffusivity, u, x, t, equations) = diffusivity
@inline diffusivity_upper_bound(diffusivity) = diffusivity

function Base.similar(coefficient::ConstantDiffusivity,
                      ::Type{NewRealT}) where {NewRealT}
    return ConstantDiffusivity(convert(NewRealT, coefficient.value))
end

function Base.similar(coefficient::SpatiallyVaryingDiffusivity,
                      ::Type{NewRealT}) where {NewRealT}
    return SpatiallyVaryingDiffusivity(coefficient.value_function,
                                       convert(NewRealT, coefficient.upper_bound))
end
