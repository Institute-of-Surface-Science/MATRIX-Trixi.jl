# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function limiter_bounds_preserving_zhang_shu!(u, lower::Union{Nothing, Real},
                                              upper::Union{Nothing, Real}, variable,
                                              mesh::AbstractMesh{2}, equations,
                                              dg::DGSEM,
                                              cache)
    @threaded for element in eachelement(dg, cache)
        u_node = get_node_vars(u, equations, dg, 1, 1, element)
        value_min = value_max = variable(u_node, equations)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            value = variable(u_node, equations)
            value_min = min(value_min, value)
            value_max = max(value_max, value)
        end

        lower_satisfied = lower === nothing || value_min >= lower
        upper_satisfied = upper === nothing || value_max <= upper
        lower_satisfied && upper_satisfied && continue

        u_mean = compute_u_mean(u, element, mesh, equations, dg, cache)
        value_mean = variable(u_mean, equations)
        theta = bounds_preserving_theta(value_min, value_max, value_mean, lower, upper)

        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
                           equations, dg, i, j, element)
        end
    end

    return nothing
end
end # @muladd
