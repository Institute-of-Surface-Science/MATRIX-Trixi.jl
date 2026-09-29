# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function limiter_bounds_preserving_zhang_shu!(u, lower::Union{Nothing, Real},
                                              upper::Union{Nothing, Real}, variable,
                                              mesh::AbstractMesh{1}, equations,
                                              dg::DGSEM,
                                              cache)
    return limiter_bounds_preserving_zhang_shu_1d!(u, lower, upper, variable,
                                                   mesh, equations, dg, cache,
                                                   Val(false))
end

function limiter_bounds_preserving_polynomial!(u, lower, upper, variable,
                                               mesh::AbstractMesh{1}, equations,
                                               dg::DGSEM, cache)
    return limiter_bounds_preserving_zhang_shu_1d!(u, lower, upper, variable,
                                                   mesh, equations, dg, cache,
                                                   Val(true))
end

function limiter_bounds_preserving_zhang_shu_1d!(u, lower, upper, variable,
                                                 mesh, equations, dg, cache,
                                                 ::Val{POLYNOMIAL}) where {POLYNOMIAL}
    @threaded for element in eachelement(dg, cache)
        if POLYNOMIAL
            value_min, value_max = bounds_preserving_polynomial_extrema(u, element,
                                                                        variable,
                                                                        equations, dg)
        else
            u_node = get_node_vars(u, equations, dg, 1, element)
            value_min = value_max = variable(u_node, equations)
            for i in eachnode(dg)
                u_node = get_node_vars(u, equations, dg, i, element)
                value = variable(u_node, equations)
                value_min = min(value_min, value)
                value_max = max(value_max, value)
            end
        end

        lower_satisfied = lower === nothing || value_min >= lower
        upper_satisfied = upper === nothing || value_max <= upper
        lower_satisfied && upper_satisfied && continue

        u_mean = compute_u_mean(u, element, mesh, equations, dg, cache)
        value_mean = variable(u_mean, equations)
        theta = bounds_preserving_theta(value_min, value_max, value_mean, lower, upper)

        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
                           equations, dg, i, element)
        end
    end

    return nothing
end

# Keep the ntuple closure outside the Polyester loop so captured array arguments
# are passed normally rather than rewritten by the threading macro.
@inline function bounds_preserving_polynomial_extrema(u, element, variable, equations,
                                                      dg)
    values = SVector(ntuple(Val(nnodes(dg))) do i
                         variable(get_node_vars(u, equations, dg, i, element),
                                  equations)
                     end)
    return bounds_preserving_polynomial_extrema(values, dg.basis)
end

# Work in the Legendre basis already used by DGSEM instead of forming a monomial
# Vandermonde matrix. Removing a constant offset preserves exactly constant states
# and reduces cancellation near a bound with a large nonzero value.
function bounds_preserving_polynomial_extrema(values::SVector{N}, basis) where {N}
    value_min, value_max = extrema(values)
    value_min == value_max && return value_min, value_max
    all(isfinite, values) || return value_min, value_max

    offset = first(values)
    centered = values .- offset
    inverse_vandermonde = basis.inverse_vandermonde_legendre
    coefficients = SVector(ntuple(Val(N)) do i
                               coefficient = zero(inverse_vandermonde[i, 1] *
                                                  centered[1])
                               for j in 1:N
                                   coefficient += inverse_vandermonde[i, j] *
                                                  centered[j]
                               end
                               # DGSEM stores orthonormal Legendre coefficients;
                               # the recurrence below uses standard P_k(1) = 1.
                               coefficient * sqrt(oftype(coefficient, i) -
                                    oftype(coefficient, 0.5))
                           end)
    derivative = bounds_preserving_legendre_derivative(coefficients)
    stationary_points, count = bounds_preserving_legendre_roots(derivative)
    one_ = one(eltype(coefficients))
    for x in (-one_, one_)
        value = offset + bounds_preserving_legendre_value(coefficients, x)
        value_min = min(value_min, value)
        value_max = max(value_max, value)
    end
    for i in 1:count
        value = offset + bounds_preserving_legendre_value(coefficients,
                                                 stationary_points[i])
        value_min = min(value_min, value)
        value_max = max(value_max, value)
    end
    return value_min, value_max
end

# dP_j/dx = sum((2k + 1) P_k(x), k = j-1, j-3, ...).
function bounds_preserving_legendre_derivative(coefficients::SVector{N}) where {N}
    return SVector(ntuple(Val(N - 1)) do i
                       coefficient = zero(eltype(coefficients))
                       for j in (i + 1):2:N
                           coefficient += coefficients[j]
                       end
                       (2 * i - 1) * coefficient
                   end)
end

# Clenshaw evaluation for the standard Legendre polynomials P_k(1) = 1.
@inline function bounds_preserving_legendre_value(coefficients::SVector{N}, x) where {N}
    next = next_next = zero(x * first(coefficients))
    for k in (N - 1):-1:0
        value = coefficients[k + 1] + (2 * k + 1) * x * next / (k + 1) -
                (k + 1) * next_next / (k + 2)
        next_next = next
        next = value
    end
    return next
end

# Derivative roots partition [-1, 1] into monotone intervals. Recursively isolate
# those intervals and bisect sign changes. This also handles degree reduction,
# repeated roots, and constants without a companion-matrix eigensolve. A repeated
# root that does not change sign is not needed for the extrema of its primitive.
function bounds_preserving_legendre_roots(coefficients::SVector{N, T}) where {N, T}
    critical_points, critical_count = bounds_preserving_legendre_roots(bounds_preserving_legendre_derivative(coefficients))
    roots = zero(SVector{N - 1, T})
    count = 0
    left = -one(T)
    value_left = bounds_preserving_legendre_value(coefficients, left)
    for i in 1:(critical_count + 1)
        right = i <= critical_count ? critical_points[i] : one(T)
        value_right = bounds_preserving_legendre_value(coefficients, right)
        if iszero(value_right)
            if right < one(T)
                count += 1
                roots = Base.setindex(roots, right, count)
            end
        elseif !iszero(value_left) && signbit(value_left) != signbit(value_right)
            root = bounds_preserving_legendre_bisect(coefficients, left, right,
                                                     value_left)
            count += 1
            roots = Base.setindex(roots, root, count)
        end
        left = right
        value_left = value_right
    end
    return roots, count
end

function bounds_preserving_legendre_roots(coefficients::SVector{1, T}) where {T}
    return SVector{0, T}(), 0
end

function bounds_preserving_legendre_roots(coefficients::SVector{2, T}) where {T}
    root = -coefficients[1] / coefficients[2]
    if isfinite(root) && -one(T) < root < one(T)
        return SVector(root), 1
    end
    return zero(SVector{1, T}), 0
end

function bounds_preserving_legendre_bisect(coefficients, left, right, value_left)
    # An absolute root error of a few ulps on [-1, 1] suffices: polynomial values
    # at stationary points are insensitive to first-order root-location errors.
    for _ in 1:(precision(typeof(left)) + 2)
        middle = left + (right - left) / 2
        (middle == left || middle == right) && return middle
        value_middle = bounds_preserving_legendre_value(coefficients, middle)
        iszero(value_middle) && return middle
        if signbit(value_middle) == signbit(value_left)
            left = middle
            value_left = value_middle
        else
            right = middle
        end
    end
    return left + (right - left) / 2
end
end # @muladd
