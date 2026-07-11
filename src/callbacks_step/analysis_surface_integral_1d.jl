# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function normal_sign_1d(direction, ::Type{T}) where {T}
    if direction == 1
        return -one(T) # x_neg
    elseif direction == 2
        return one(T) # x_pos
    else
        throw(ArgumentError("Invalid 1D boundary direction $direction"))
    end
end

function analyze(surface_variable::AnalysisSurfaceIntegral{Variable}, du, u, t,
                 mesh::TreeMesh{1}, equations::AbstractEquationsParabolic,
                 dg::DGSEM, cache,
                 semi::SemidiscretizationParabolic) where {
                                                           Component,
                                                           Variable <:
                                                           NormalParabolicFlux{Component}
                                                           }
    @unpack variable, boundary_symbols = surface_variable
    @unpack surface_flux_values = cache.elements
    @unpack neighbor_ids, n_boundaries_per_direction = cache.boundaries

    if !(1 <= Component <= nvariables(equations))
        throw(ArgumentError("Requested parabolic flux component $Component, but " *
                            "$(typeof(equations)) has $(nvariables(equations)) variable(s)"))
    end

    boundary_conditions = semi.boundary_conditions
    boundary_conditions isa NamedTuple ||
        throw(ArgumentError("NormalParabolicFlux requires named nonperiodic boundary conditions"))

    boundary_names = keys(boundary_conditions)
    counts = n_boundaries_per_direction
    lasts = accumulate(+, counts)
    firsts = lasts - counts .+ 1

    result = zero(eltype(u))
    for boundary_symbol in boundary_symbols
        direction = findfirst(isequal(boundary_symbol), boundary_names)
        if isnothing(direction)
            throw(ArgumentError("Unknown boundary symbol `$boundary_symbol`. Available " *
                                "boundaries are $(Tuple(boundary_names))"))
        end

        count = counts[direction]
        count == 0 && continue

        normal = normal_sign_1d(direction, eltype(u))
        for boundary in firsts[direction]:lasts[direction]
            element = neighbor_ids[boundary]
            # This boundary-condition-adjusted flux is the value used by the DG divergence.
            flux_component = surface_flux_values[Component, direction, element]
            result += variable.factor * normal * flux_component
        end
    end

    if mpi_isparallel()
        result = MPI.Allreduce!(Ref(result), +, mpi_comm())[]
    end

    return result
end
end # muladd
