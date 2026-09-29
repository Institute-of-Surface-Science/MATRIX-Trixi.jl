# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function limiter_bounds_preserving_polynomial!(u, lower, upper, variable,
                                               mesh::Union{AbstractMesh{2},
                                                           AbstractMesh{3}},
                                               equations, dg::DGSEM, cache)
    transform = bounds_preserving_bernstein_matrix(dg.basis.nodes)
    # Each thread owns its scratch arrays; no storage is shared between limiter calls.
    scalar_type = typeof(variable(zero(SVector{nvariables(equations), eltype(u)}),
                                  equations))
    T = promote_type(eltype(u), scalar_type, eltype(transform))
    shape = ntuple(_ -> nnodes(dg), ndims(mesh))
    workspaces = [(buffers = [Array{T}(undef, shape) for _ in 1:(4 * ndims(mesh) + 2)],
                   line = Vector{T}(undef, nnodes(dg)))
                  for _ in 1:Threads.maxthreadid()]

    @threaded for element in eachelement(dg, cache)
        bounds_preserving_bernstein_element!(u, element, lower, upper, variable,
                                             mesh, equations, dg, cache, transform,
                                             workspaces[Threads.threadid()])
    end
    return nothing
end

function bounds_preserving_bernstein_element!(u, element, lower, upper, variable,
                                              mesh, equations, dg, cache, transform,
                                              workspace)
    values = first(workspace.buffers)
    for node in CartesianIndices(values)
        values[node] = variable(get_node_vars(u, equations, dg, Tuple(node)...,
                                              element),
                                equations)
    end
    value_min, value_max = bounds_preserving_bernstein_bounds!(workspace, transform,
                                                               lower, upper)
    lower_satisfied = lower === nothing || value_min >= lower
    upper_satisfied = upper === nothing || value_max <= upper
    lower_satisfied && upper_satisfied && return nothing

    u_mean = compute_u_mean(u, element, mesh, equations, dg, cache)
    value_mean = variable(u_mean, equations)
    theta = bounds_preserving_theta(value_min, value_max, value_mean, lower, upper)
    for node in CartesianIndices(values)
        u_node = get_node_vars(u, equations, dg, Tuple(node)..., element)
        set_node_vars!(u, theta * u_node + (1 - theta) * u_mean, equations, dg,
                       Tuple(node)..., element)
    end
    return nothing
end

# Construct each Lagrange cardinal polynomial directly in Bernstein form on [-1, 1].
# Multiplication by (x - nodes[k])/(nodes[j] - nodes[k]) increases the degree by one.
# This avoids an additional Vandermonde inversion and works for Gauss and Lobatto nodes.
function bounds_preserving_bernstein_matrix(nodes)
    n = length(nodes)
    transform = zeros(eltype(nodes), n, n)
    for j in 1:n
        transform[1, j] = 1
        degree = 0
        for k in 1:n
            k == j && continue
            degree += 1
            left = (-one(eltype(nodes)) - nodes[k]) / (nodes[j] - nodes[k])
            right = (one(eltype(nodes)) - nodes[k]) / (nodes[j] - nodes[k])
            for i in degree:-1:0
                fraction = i / oftype(left, degree)
                previous = i == 0 ? zero(left) : transform[i, j]
                transform[i + 1, j] = (1 - fraction) * left * transform[i + 1, j] +
                                      fraction * right * previous
            end
        end
    end
    return transform
end

function bounds_preserving_bernstein_bounds!(workspace, transform, lower, upper)
    values, temporary = workspace.buffers[1], workspace.buffers[2]
    nodal_min, nodal_max = extrema(values)
    nodal_min == nodal_max && return nodal_min, nodal_max
    all(isfinite, values) || return nodal_min, nodal_max

    # Preserve constants exactly and reduce cancellation near a nonzero bound.
    offset = first(values)
    values .-= offset
    for dimension in 1:ndims(values)
        bounds_preserving_bernstein_transform!(temporary, transform, values, dimension)
        values, temporary = temporary, values
    end
    # An overflowing transform must not silently certify an element as bounded.
    all(isfinite, values) || return oftype(offset, -Inf), oftype(offset, Inf)
    lower_centered = lower === nothing ? nothing : lower - offset
    upper_centered = upper === nothing ? nothing : upper - offset
    value_min, value_max = bounds_preserving_bernstein_subdivide!(values, workspace,
                                                                  lower_centered,
                                                                  upper_centered, 1)
    return min(nodal_min, offset + value_min), max(nodal_max, offset + value_max)
end

function bounds_preserving_bernstein_transform!(result, transform, values, dimension)
    for node in CartesianIndices(result)
        value = zero(eltype(result))
        for k in axes(transform, 2)
            source = Base.setindex(Tuple(node), k, dimension)
            value += transform[node[dimension], k] * values[source...]
        end
        result[node] = value
    end
    return nothing
end

# Tensor-product Bernstein basis functions are nonnegative and sum to one, hence
# their coefficient range encloses the polynomial throughout a box. Subdivision
# tightens the enclosure without relying on sampling or locating stationary points.
# See Ray & Nataraj (2010), Reliable Computing 14, 117-137, eq. (4):
# https://interval.louisiana.edu/reliable-computing-journal/volume-14/reliable-computing-14-pp-117-137.pdf
function bounds_preserving_bernstein_subdivide!(coefficients, workspace, lower, upper,
                                                depth)
    value_min, value_max = extrema(coefficients)
    lower_satisfied = lower === nothing || value_min >= lower
    upper_satisfied = upper === nothing || value_max <= upper
    lower_satisfied && upper_satisfied && return value_min, value_max
    # At most two bisections per coordinate: 16 subboxes in 2D and 64 in 3D.
    # At this budget, keep the enclosing coefficients, never replace them by samples.
    depth > 2 * ndims(coefficients) && return value_min, value_max

    dimension = mod1(depth, ndims(coefficients))
    left, right = workspace.buffers[2 * depth + 1], workspace.buffers[2 * depth + 2]
    bounds_preserving_bernstein_split!(left, right, coefficients, workspace.line,
                                       dimension)
    left_min, left_max = bounds_preserving_bernstein_subdivide!(left, workspace,
                                                                lower, upper, depth + 1)
    right_min, right_max = bounds_preserving_bernstein_subdivide!(right, workspace,
                                                                  lower, upper,
                                                                  depth + 1)
    return min(left_min, right_min), max(left_max, right_max)
end

# de Casteljau subdivision at the midpoint, independently along each tensor line.
function bounds_preserving_bernstein_split!(left, right, coefficients, line, dimension)
    n = length(line)
    lines = CartesianIndices(Base.setindex(size(coefficients), 1, dimension))
    for node in lines
        for i in 1:n
            line[i] = coefficients[Base.setindex(Tuple(node), i, dimension)...]
        end
        left[Tuple(node)...] = first(line)
        right[Base.setindex(Tuple(node), n, dimension)...] = last(line)
        for level in 1:(n - 1)
            for i in 1:(n - level)
                line[i] = line[i] / 2 + line[i + 1] / 2
            end
            left[Base.setindex(Tuple(node), level + 1, dimension)...] = first(line)
            right[Base.setindex(Tuple(node), n - level, dimension)...] = line[n - level]
        end
    end
    return nothing
end
end # @muladd
