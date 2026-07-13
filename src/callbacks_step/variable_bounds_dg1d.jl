function local_variable_extrema(variable, u, mesh::AbstractMesh{1}, equations,
                                dg::DGSEM, cache)
    ValueT = variable_value_type(variable, equations, dg)
    RealT = promote_type(real(dg), ValueT)
    value_min = typemax(RealT)
    value_max = typemin(RealT)
    finite_count = 0
    nonfinite_count = 0

    @batch reduction=((min, value_min), (max, value_max), (+, finite_count),
                      (+, nonfinite_count)) for element in eachelement(dg, cache)
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            value = variable(u_node, equations)
            value isa Real ||
                throw(ArgumentError("variables monitored by VariableBoundsCallback " *
                                    "must return real numbers"))
            value = convert(RealT, value)

            if isfinite(value)
                value_min = Base.min(value_min, value)
                value_max = Base.max(value_max, value)
                finite_count += 1
            else
                nonfinite_count += 1
            end
        end
    end

    return value_min, value_max, finite_count, nonfinite_count
end
