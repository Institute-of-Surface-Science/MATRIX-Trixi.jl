# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    VariableBound(name, variable;
                  lower=nothing,
                  upper=nothing,
                  abstol=0,
                  reltol=0)

Describe admissible bounds for a scalar quantity. The callable `variable` must have the
signature

    variable(u_node, equations) -> Real

where `u_node` is the local state vector. The return type must be a concrete `Real` type.
At least one of `lower` and `upper` must be provided. An unused side of the interval is
represented by `nothing`; specified bounds and tolerances must be finite.
"""
struct VariableBound{Variable, Lower, Upper, AbsTol, RelTol}
    name::Symbol
    variable::Variable
    lower::Lower
    upper::Upper
    abstol::AbsTol
    reltol::RelTol

    function VariableBound(name::Symbol, variable, lower, upper, abstol, reltol)
        lower === nothing && upper === nothing &&
            throw(ArgumentError("at least one of `lower` and `upper` must be specified"))

        if lower !== nothing && !(lower isa Real)
            throw(ArgumentError("`lower` must be a real number or `nothing`"))
        end
        if lower !== nothing && !isfinite(lower)
            throw(ArgumentError("`lower` must be finite"))
        end
        if upper !== nothing && !(upper isa Real)
            throw(ArgumentError("`upper` must be a real number or `nothing`"))
        end
        if upper !== nothing && !isfinite(upper)
            throw(ArgumentError("`upper` must be finite"))
        end
        if lower !== nothing && upper !== nothing && lower > upper
            throw(ArgumentError("`lower` must not be greater than `upper`"))
        end

        abstol isa Real || throw(ArgumentError("`abstol` must be a real number"))
        isfinite(abstol) || throw(ArgumentError("`abstol` must be finite"))
        abstol >= 0 || throw(ArgumentError("`abstol` must be non-negative"))
        reltol isa Real || throw(ArgumentError("`reltol` must be a real number"))
        isfinite(reltol) || throw(ArgumentError("`reltol` must be finite"))
        reltol >= 0 || throw(ArgumentError("`reltol` must be non-negative"))

        return new{typeof(variable), typeof(lower), typeof(upper), typeof(abstol),
                   typeof(reltol)}(name, variable, lower, upper, abstol, reltol)
    end
end

function VariableBound(name::Symbol, variable;
                       lower = nothing,
                       upper = nothing,
                       abstol = 0,
                       reltol = 0)
    return VariableBound(name, variable, lower, upper, abstol, reltol)
end

"""
    VariableBoundsResult

Extrema and raw bound violations computed by [`VariableBoundsCallback`](@ref). A result
is considered violated when either bound is exceeded beyond its tolerance or at least one
sample is not finite; see [`isviolated`](@ref).
"""
struct VariableBoundsResult{RealT <: Real}
    minimum::RealT
    maximum::RealT
    lower_violation::RealT
    upper_violation::RealT
    finite_count::Int
    nonfinite_count::Int
    lower_violated::Bool
    upper_violated::Bool
end

"""
    isviolated(result::VariableBoundsResult)

Return whether `result` contains a bound violation or a nonfinite sample.
"""
@inline function isviolated(result::VariableBoundsResult)
    return result.lower_violated || result.upper_violated ||
           result.nonfinite_count > 0
end

mutable struct VariableBoundsCallback{Bounds, Results, WorstLower, WorstUpper,
                                      NonfiniteTotals}
    const bounds::Bounds
    const interval::Int
    const check_initial::Bool
    const check_final::Bool
    const action::Symbol

    const save_analysis::Bool
    const output_directory::String
    const analysis_filename::String

    last_results::Results
    worst_lower_violation::WorstLower
    worst_upper_violation::WorstUpper
    nonfinite_totals::NonfiniteTotals

    checks_performed::Int
    violations_detected::Int
end

"""
    VariableBoundsCallback(semi; bounds, interval=1,
                           check_initial=true, check_final=true,
                           action=:warn, save_analysis=false,
                           output_directory="out",
                           analysis_filename="variable_bounds.dat")

Monitor the global nodal extrema of custom scalar variables in a DGSEM solution. Each
entry of `bounds` must be a [`VariableBound`](@ref). Checks are performed after every
`interval` accepted steps, with `interval=0` disabling periodic checks.

Available actions are `:record`, `:warn`, and `:terminate`. The callback is diagnostic
only and never modifies the numerical solution. File output is enabled with
`save_analysis=true` and uses one row per variable and check.

This callback supports DGSEM semidiscretizations on one-, two-, and three-dimensional
meshes. DGMulti support is not currently available.
"""
# This is the convenience constructor that gets called from the elixirs
function VariableBoundsCallback(semi::AbstractSemidiscretization;
                                kwargs...)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    return VariableBoundsCallback(mesh, equations, solver, cache; kwargs...)
end

# This is the actual constructor
function VariableBoundsCallback(mesh::Union{AbstractMesh{1}, AbstractMesh{2},
                                            AbstractMesh{3}},
                                equations, solver::DGSEM, cache;
                                bounds,
                                interval = 1,
                                check_initial = true,
                                check_final = true,
                                action = :warn,
                                save_analysis = false,
                                output_directory = "out",
                                analysis_filename = "variable_bounds.dat")
    interval isa Integer ||
        throw(ArgumentError("`interval` must be a non-negative integer"))
    interval >= 0 || throw(ArgumentError("`interval` must be non-negative"))
    action in (:record, :warn, :terminate) ||
        throw(ArgumentError("`action` must be `:record`, `:warn`, or `:terminate`"))

    bounds = Tuple(bounds)
    all(bound -> bound isa VariableBound, bounds) ||
        throw(ArgumentError("all entries of `bounds` must be `VariableBound`s"))
    names = map(bound -> bound.name, bounds)
    allunique(names) ||
        throw(ArgumentError("all variable-bound names must be unique"))

    uEltype = eltype(cache.elements)
    u_node = SVector(ntuple(_ -> zero(uEltype), nvariables(equations)))
    for bound in bounds
        applicable(bound.variable, u_node, equations) ||
            throw(ArgumentError("variable `$(bound.name)` must be callable as variable(u_node, equations)"))
        variable_value_type(bound.variable, equations, uEltype)
    end

    last_results = initial_variable_bounds_results(bounds, equations, uEltype)
    result_values = values(last_results)
    worst_lower = map(result -> zero(result.lower_violation), result_values)
    worst_upper = map(result -> zero(result.upper_violation), result_values)
    nonfinite_totals = map(_ -> 0, result_values)

    CallbackT = VariableBoundsCallback{typeof(bounds), typeof(last_results),
                                       typeof(worst_lower), typeof(worst_upper),
                                       typeof(nonfinite_totals)}
    variable_bounds_callback = CallbackT(bounds, Int(interval), check_initial,
                                         check_final,
                                         action, save_analysis,
                                         String(output_directory),
                                         String(analysis_filename), last_results,
                                         worst_lower,
                                         worst_upper, nonfinite_totals, 0, 0)

    condition = function (u, t, integrator)
        interval_ = variable_bounds_callback.interval
        periodic_check = interval_ > 0 &&
                         integrator.stats.naccept > 0 &&
                         iszero(integrator.stats.naccept % interval_)
        final_check = variable_bounds_callback.check_final && isfinished(integrator)
        return periodic_check || final_check
    end

    return DiscreteCallback(condition, variable_bounds_callback;
                            save_positions = (false, false),
                            initialize = initialize!)
end

function VariableBoundsCallback(mesh, equations, solver, cache; kwargs...)
    throw(ArgumentError("VariableBoundsCallback requires a one-, two-, or " *
                        "three-dimensional mesh with a DGSEM solver"))
end

function variable_value_type(variable, equations,
                             ::Type{uEltype}) where {uEltype <: Real}
    NodeT = SVector{nvariables(equations), uEltype}
    ValueT = Base.promote_op(variable, NodeT, typeof(equations))
    if !isconcretetype(ValueT) || !(ValueT <: Real)
        throw(ArgumentError("variables monitored by VariableBoundsCallback must " *
                            "return a concrete real type"))
    end
    return ValueT
end

function variable_bounds_result_type(bound, ValueT)
    LowerT = bound.lower === nothing ? ValueT : typeof(bound.lower)
    UpperT = bound.upper === nothing ? ValueT : typeof(bound.upper)
    return promote_type(ValueT, LowerT, UpperT, typeof(bound.abstol),
                        typeof(bound.reltol))
end

function initial_variable_bounds_results(bounds, equations,
                                         ::Type{uEltype}) where {uEltype <: Real}
    results = map(bounds) do bound
        ValueT = variable_value_type(bound.variable, equations, uEltype)
        ExtremaT = promote_type(uEltype, ValueT)
        ResultT = variable_bounds_result_type(bound, ExtremaT)
        nan = convert(ResultT, NaN)
        zero_value = zero(ResultT)
        VariableBoundsResult(nan, nan, zero_value, zero_value, 0, 0, false, false)
    end
    names = map(bound -> bound.name, bounds)
    return NamedTuple{names}(results)
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:VariableBoundsCallback})
    @nospecialize cb # reduce precompilation time

    callback = cb.affect!
    print(io, "VariableBoundsCallback(variables=", length(callback.bounds),
          ", interval=", callback.interval, ", action=:", callback.action, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:VariableBoundsCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!
        setup = Pair{String, Any}["interval" => callback.interval,
                                  "check initial state" => callback.check_initial ?
                                                           "yes" : "no",
                                  "check final state" => callback.check_final ? "yes" :
                                                         "no",
                                  "action" => callback.action,
                                  "save analysis" => callback.save_analysis ? "yes" :
                                                     "no"]
        for (index, bound) in enumerate(callback.bounds)
            lower = something(bound.lower, "-Inf")
            upper = something(bound.upper, "Inf")
            push!(setup, "variable $index" => "$(bound.name) in [$lower, $upper]")
        end
        if callback.save_analysis
            push!(setup, "filename" => callback.analysis_filename)
            push!(setup,
                  "output directory" => abspath(normpath(callback.output_directory)))
        end
        summary_box(io, "VariableBoundsCallback", setup)
    end
    return nothing
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode, t,
                     integrator) where {Condition, Affect! <: VariableBoundsCallback}
    callback = cb.affect!
    result_names = keys(callback.last_results)
    reset_results = map(values(callback.last_results)) do result
        RealT = typeof(result.minimum)
        nan = convert(RealT, NaN)
        zero_value = zero(RealT)
        VariableBoundsResult(nan, nan, zero_value, zero_value, 0, 0, false, false)
    end
    callback.last_results = NamedTuple{result_names}(reset_results)
    callback.worst_lower_violation = map(result -> zero(result.lower_violation),
                                         values(callback.last_results))
    callback.worst_upper_violation = map(result -> zero(result.upper_violation),
                                         values(callback.last_results))
    callback.nonfinite_totals = map(_ -> 0, values(callback.last_results))
    callback.checks_performed = 0
    callback.violations_detected = 0

    if callback.save_analysis && mpi_isroot()
        mkpath(callback.output_directory)
        open(joinpath(callback.output_directory, callback.analysis_filename), "w") do io
            println(io,
                    "# iter time name minimum maximum lower upper lower_violation " *
                    "upper_violation lower_tolerance upper_tolerance finite_count " *
                    "nonfinite_count violated")
        end
    end

    if callback.check_initial
        check_variable_bounds!(callback, u_ode, t, 0, integrator)
    end
    return nothing
end

function (callback::VariableBoundsCallback)(integrator)
    check_variable_bounds!(callback, integrator.u, integrator.t,
                           integrator.stats.naccept, integrator)
    return nothing
end

function check_variable_bounds!(callback::VariableBoundsCallback, u_ode, t, iter,
                                integrator)
    results = @trixi_timeit timer() "variable bounds" evaluate_variable_bounds(u_ode,
                                                                               integrator.p,
                                                                               callback.bounds)
    callback.last_results = results
    callback.checks_performed += 1

    result_values = values(results)
    callback.worst_lower_violation = map(Base.max,
                                         callback.worst_lower_violation,
                                         map(result -> result.lower_violation,
                                             result_values))
    callback.worst_upper_violation = map(Base.max,
                                         callback.worst_upper_violation,
                                         map(result -> result.upper_violation,
                                             result_values))
    callback.nonfinite_totals = map(+,
                                    callback.nonfinite_totals,
                                    map(result -> result.nonfinite_count,
                                        result_values))

    violation_detected = any(isviolated, result_values)
    callback.violations_detected += violation_detected

    if callback.save_analysis && mpi_isroot()
        write_variable_bounds(callback, results, iter, t)
    end

    if violation_detected && callback.action !== :record
        mpi_isroot() && report_variable_bounds_violations(callback, results, iter, t)
        if callback.action === :terminate
            terminate!(integrator)
        end
    end

    if isfinished(integrator) && mpi_isroot()
        print_variable_bounds_summary(callback)
    end

    # avoid re-evaluating possible FSAL stages
    derivative_discontinuity!(integrator, false)
    return nothing
end

function evaluate_variable_bounds(u_ode, semi, bounds)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)
    results = evaluate_variable_bounds(bounds, u, mesh, equations, solver, cache)
    names = map(bound -> bound.name, bounds)
    return NamedTuple{names}(results)
end

function evaluate_variable_bounds(bounds::NTuple{N, Any}, u, mesh, equations, solver,
                                  cache) where {N}
    current = evaluate_variable_bound(first(bounds), u, mesh, equations, solver, cache)
    remaining = evaluate_variable_bounds(Base.tail(bounds), u, mesh, equations, solver,
                                         cache)
    return (current, remaining...)
end

function evaluate_variable_bounds(::Tuple{}, u, mesh, equations, solver, cache)
    return ()
end

function local_variable_extrema(variable, u, mesh::AbstractMesh{NDIMS}, equations,
                                dg::DGSEM, cache) where {NDIMS}
    uEltype = eltype(u)
    ValueT = variable_value_type(variable, equations, uEltype)
    RealT = promote_type(uEltype, ValueT)
    value_min = typemax(RealT)
    value_max = typemin(RealT)
    finite_count = 0
    nonfinite_count = 0

    if isbitstype(RealT)
        @batch reduction=((min, value_min), (max, value_max), (+, finite_count),
                          (+, nonfinite_count)) for element in eachelement(dg, cache)
            element_extrema = variable_extrema_at_element(variable, u, element, mesh,
                                                          equations, dg, RealT)
            value_min = Base.min(value_min, element_extrema[1])
            value_max = Base.max(value_max, element_extrema[2])
            finite_count += element_extrema[3]
            nonfinite_count += element_extrema[4]
        end
    else
        for element in eachelement(dg, cache)
            element_extrema = variable_extrema_at_element(variable, u, element, mesh,
                                                          equations, dg, RealT)
            value_min = Base.min(value_min, element_extrema[1])
            value_max = Base.max(value_max, element_extrema[2])
            finite_count += element_extrema[3]
            nonfinite_count += element_extrema[4]
        end
    end

    return value_min, value_max, finite_count, nonfinite_count
end

function evaluate_variable_bound(bound::VariableBound, u, mesh, equations, solver,
                                 cache)
    value_min, value_max, finite_count, nonfinite_count = local_variable_extrema(bound.variable,
                                                                                 u,
                                                                                 mesh,
                                                                                 equations,
                                                                                 solver,
                                                                                 cache)

    if mpi_isparallel()
        value_min = MPI.Allreduce!(Ref(value_min), Base.min, mpi_comm())[]
        value_max = MPI.Allreduce!(Ref(value_max), Base.max, mpi_comm())[]
        finite_count = MPI.Allreduce!(Ref(finite_count), +, mpi_comm())[]
        nonfinite_count = MPI.Allreduce!(Ref(nonfinite_count), +, mpi_comm())[]
    end

    if finite_count == 0
        value_min = oftype(value_min, NaN)
        value_max = oftype(value_max, NaN)
    end

    return make_bounds_result(bound, value_min, value_max, finite_count,
                              nonfinite_count)
end

function make_bounds_result(bound::VariableBound, value_min, value_max, finite_count,
                            nonfinite_count)
    ValueT = promote_type(typeof(value_min), typeof(value_max))
    RealT = variable_bounds_result_type(bound, ValueT)
    minimum = convert(RealT, value_min)
    maximum = convert(RealT, value_max)
    zero_value = zero(RealT)

    lower_violation, lower_violated = lower_bounds_status(bound, minimum, finite_count,
                                                          zero_value)
    upper_violation, upper_violated = upper_bounds_status(bound, maximum, finite_count,
                                                          zero_value)

    return VariableBoundsResult(minimum, maximum, lower_violation, upper_violation,
                                finite_count, nonfinite_count, lower_violated,
                                upper_violated)
end

@inline function lower_bounds_status(bound, minimum, finite_count, zero_value)
    if bound.lower === nothing || finite_count == 0
        return zero_value, false
    end

    lower = convert(typeof(minimum), bound.lower)
    violation = Base.max(zero_value, lower - minimum)
    tolerance = convert(typeof(minimum), bound.abstol) +
                convert(typeof(minimum), bound.reltol) *
                Base.max(abs(lower), abs(minimum))
    return violation, violation > tolerance
end

@inline function upper_bounds_status(bound, maximum, finite_count, zero_value)
    if bound.upper === nothing || finite_count == 0
        return zero_value, false
    end

    upper = convert(typeof(maximum), bound.upper)
    violation = Base.max(zero_value, maximum - upper)
    tolerance = convert(typeof(maximum), bound.abstol) +
                convert(typeof(maximum), bound.reltol) *
                Base.max(abs(upper), abs(maximum))
    return violation, violation > tolerance
end

function bound_tolerances(bound, result)
    RealT = typeof(result.minimum)
    if bound.lower === nothing || result.finite_count == 0
        lower_tolerance = convert(RealT, NaN)
    else
        lower = convert(RealT, bound.lower)
        lower_tolerance = convert(RealT, bound.abstol) +
                          convert(RealT, bound.reltol) *
                          Base.max(abs(lower), abs(result.minimum))
    end
    if bound.upper === nothing || result.finite_count == 0
        upper_tolerance = convert(RealT, NaN)
    else
        upper = convert(RealT, bound.upper)
        upper_tolerance = convert(RealT, bound.abstol) +
                          convert(RealT, bound.reltol) *
                          Base.max(abs(upper), abs(result.maximum))
    end
    return lower_tolerance, upper_tolerance
end

function write_variable_bounds(callback, results, iter, t)
    open(joinpath(callback.output_directory, callback.analysis_filename), "a") do io
        for (bound, result) in zip(callback.bounds, values(results))
            lower = something(bound.lower, -Inf)
            upper = something(bound.upper, Inf)
            lower_tolerance, upper_tolerance = bound_tolerances(bound, result)
            println(io, iter, " ", t, " ", bound.name, " ", result.minimum, " ",
                    result.maximum, " ", lower, " ", upper, " ",
                    result.lower_violation, " ", result.upper_violation, " ",
                    lower_tolerance, " ", upper_tolerance, " ", result.finite_count,
                    " ", result.nonfinite_count, " ", isviolated(result))
        end
    end
    return nothing
end

function report_variable_bounds_violations(callback, results, iter, t)
    for (bound, result) in zip(callback.bounds, values(results))
        isviolated(result) || continue
        lower = something(bound.lower, -Inf)
        upper = something(bound.upper, Inf)
        lower_tolerance, upper_tolerance = bound_tolerances(bound, result)
        variable = bound.name
        minimum = result.minimum
        maximum = result.maximum
        admissible_interval = (lower, upper)
        lower_violation = result.lower_violation
        upper_violation = result.upper_violation
        nonfinite_count = result.nonfinite_count
        action = callback.action
        @warn("Variable bound violated", variable, minimum, maximum,
              admissible_interval, lower_violation, lower_tolerance,
              upper_violation, upper_tolerance, nonfinite_count, time=t,
              accepted_step=iter, action)
    end
    return nothing
end

function print_variable_bounds_summary(callback)
    println("-"^72)
    println("Variable bounds summary")
    println("-"^72)
    for (index, (bound, result)) in enumerate(zip(callback.bounds,
                                                  values(callback.last_results)))
        println(bound.name, ":")
        @printf("  minimum at final check:       % .8e\n", result.minimum)
        @printf("  maximum at final check:       % .8e\n", result.maximum)
        @printf("  worst lower violation:        % .8e\n",
                callback.worst_lower_violation[index])
        @printf("  worst upper violation:        % .8e\n",
                callback.worst_upper_violation[index])
        println("  nonfinite values observed:    ", callback.nonfinite_totals[index])
    end
    println("-"^72)
    return nothing
end

function (cb::DiscreteCallback{Condition, Affect!})(sol) where {Condition,
                                                                Affect! <:
                                                                VariableBoundsCallback}
    callback = cb.affect!
    return evaluate_variable_bounds(sol.u[end], sol.prob.p, callback.bounds)
end

include("variable_bounds_dg1d.jl")
include("variable_bounds_dg2d.jl")
include("variable_bounds_dg3d.jl")
end # @muladd
