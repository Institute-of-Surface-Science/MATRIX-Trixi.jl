# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    BoundsPreservingLimiterZhangShu(; lower, upper, variables, polynomial_bounds=false)

The fully-discrete bounds-preserving limiter of
- Zhang, Shu (2010)
  On maximum-principle-satisfying high order schemes for scalar conservation laws
  [doi: 10.1016/j.jcp.2010.03.001](https://doi.org/10.1016/j.jcp.2010.03.001)

The limiter is applied to all real-valued scalar `variables` in their given order. The tuples
`lower` and `upper` contain the associated bounds; use `nothing` for an unused side.
For each variable, at least one finite bound must be specified. The order can affect the
result for nonlinear variables.

Nodal states outside an interval are scaled towards their element mean. Thus, the
limiter preserves the element mean of every conservative variable. A bound can only be
enforced when the corresponding scalar variable is affine in the conservative variables,
or when suitable convexity or concavity assumptions hold. If the value at an element mean
is outside its bounds, the element is collapsed to that mean to preserve conservation; a
[`VariableBoundsCallback`](@ref) can be used to diagnose the remaining violation.

Set `polynomial_bounds=true` for affine scalar variables in one-, two-, or
three-dimensional DGSEM solutions to also enforce the bounds between nodes. In 1D,
this mode evaluates polynomial extrema, including stationary points. In 2D and 3D,
it encloses the tensor-product polynomial using Bernstein coefficients and midpoint
subdivision. These conservative enclosures can cause additional limiting compared
with exact extrema. Both approaches scale the conservative state about its physical
element mean, so spatial interpolation remains bounded up to roundoff when the mean
is admissible. They do not bound nonlinear functions of the interpolated state or
dense time interpolation. The default mode only checks DG nodes.

Pass this limiter as the solve-level `step_limiter` with implicit OrdinaryDiffEq
algorithms such as `TRBDF2`, or as `stage_limiter` with compatible explicit Runge-Kutta
algorithms. Solve-level step limiting requires OrdinaryDiffEqCore v4.15.3 or newer
to refresh FSAL derivatives after modifying an accepted endpoint. The per-algorithm
`step_limiter!` and `stage_limiter!` fields are deprecated compatibility syntax.
"""
struct BoundsPreservingLimiterZhangShu{N, Lower, Upper, Variables}
    lower::Lower
    upper::Upper
    variables::Variables
    polynomial_bounds::Bool
end

function BoundsPreservingLimiterZhangShu(lower::Tuple, upper::Tuple, variables::Tuple;
                                         polynomial_bounds::Bool = false)
    n_variables = length(variables)
    n_variables > 0 || throw(ArgumentError("at least one variable must be specified"))
    length(lower) == n_variables ||
        throw(ArgumentError("`lower`, `upper`, and `variables` must have the same length"))
    length(upper) == n_variables ||
        throw(ArgumentError("`lower`, `upper`, and `variables` must have the same length"))

    for (lower_bound, upper_bound) in zip(lower, upper)
        lower_bound === nothing && upper_bound === nothing &&
            throw(ArgumentError("at least one bound must be specified for each variable"))

        if lower_bound !== nothing
            lower_bound isa Real ||
                throw(ArgumentError("lower bounds must be real numbers or `nothing`"))
            isfinite(lower_bound) || throw(ArgumentError("lower bounds must be finite"))
        end
        if upper_bound !== nothing
            upper_bound isa Real ||
                throw(ArgumentError("upper bounds must be real numbers or `nothing`"))
            isfinite(upper_bound) || throw(ArgumentError("upper bounds must be finite"))
        end
        if lower_bound !== nothing && upper_bound !== nothing &&
           lower_bound > upper_bound
            throw(ArgumentError("lower bounds must not be greater than upper bounds"))
        end
    end

    return BoundsPreservingLimiterZhangShu{n_variables, typeof(lower), typeof(upper),
                                           typeof(variables)}(lower, upper, variables,
                                                              polynomial_bounds)
end

function BoundsPreservingLimiterZhangShu(; lower, upper, variables,
                                         polynomial_bounds::Bool = false)
    return BoundsPreservingLimiterZhangShu(Tuple(lower), Tuple(upper), Tuple(variables);
                                           polynomial_bounds)
end

function (limiter!::BoundsPreservingLimiterZhangShu)(u_ode, integrator,
                                                     semi::AbstractSemidiscretization,
                                                     t)
    u = wrap_array(u_ode, semi)
    @trixi_timeit timer() "Zhang-Shu bounds-preserving limiter" begin
        limiter_bounds_preserving_zhang_shu!(u, limiter!.lower, limiter!.upper,
                                             limiter!.variables,
                                             mesh_equations_solver_cache(semi)...;
                                             polynomial_bounds = limiter!.polynomial_bounds)
    end

    return nothing
end

# Iterate over tuples in a type-stable way. The order matters for nonlinear variables since
# limiting one variable can alter the nodal values of variables limited earlier.
function limiter_bounds_preserving_zhang_shu!(u, lower::NTuple{N, Any},
                                              upper::NTuple{N, Any},
                                              variables::NTuple{N, Any}, mesh,
                                              equations,
                                              solver, cache;
                                              polynomial_bounds = false) where {N}
    if polynomial_bounds
        limiter_bounds_preserving_polynomial!(u, first(lower), first(upper),
                                              first(variables), mesh, equations,
                                              solver, cache)
    else
        limiter_bounds_preserving_zhang_shu!(u, first(lower), first(upper),
                                             first(variables), mesh, equations, solver,
                                             cache)
    end
    limiter_bounds_preserving_zhang_shu!(u, Base.tail(lower), Base.tail(upper),
                                         Base.tail(variables), mesh, equations, solver,
                                         cache; polynomial_bounds)
    return nothing
end

function limiter_bounds_preserving_zhang_shu!(u, ::Tuple{}, ::Tuple{}, ::Tuple{}, mesh,
                                              equations, solver, cache;
                                              polynomial_bounds = false)
    return nothing
end

function limiter_bounds_preserving_polynomial!(u, lower, upper, variable, mesh,
                                               equations, solver, cache)
    throw(ArgumentError("polynomial bounds require a one-, two-, or three-dimensional DGSEM solution"))
end

@inline function bounds_preserving_theta(value_min, value_max, value_mean, lower, upper)
    LowerT = lower === nothing ? typeof(value_mean) : typeof(lower)
    UpperT = upper === nothing ? typeof(value_mean) : typeof(upper)
    ThetaT = promote_type(typeof(value_min), typeof(value_max), typeof(value_mean),
                          LowerT,
                          UpperT)
    theta = one(ThetaT)
    value_mean_ = convert(ThetaT, value_mean)

    if lower !== nothing
        lower_ = convert(ThetaT, lower)
        value_mean_ < lower_ && return zero(ThetaT)
        if value_min < lower_
            theta = min(theta,
                        (value_mean_ - lower_) /
                        (value_mean_ - convert(ThetaT, value_min)))
        end
    end

    if upper !== nothing
        upper_ = convert(ThetaT, upper)
        value_mean_ > upper_ && return zero(ThetaT)
        if value_max > upper_
            theta = min(theta,
                        (upper_ - value_mean_) /
                        (convert(ThetaT, value_max) - value_mean_))
        end
    end

    return max(zero(ThetaT), theta)
end

include("bounds_preserving_zhang_shu_dg1d.jl")
include("bounds_preserving_zhang_shu_dg2d.jl")
include("bounds_preserving_zhang_shu_dg3d.jl")
include("bounds_preserving_zhang_shu_bernstein.jl")
end # @muladd
