function local_variable_extrema(variable, u, mesh::AbstractMesh{2}, equations,
                                dg::DGSEM, cache)
    ValueT = variable_value_type(variable, equations, dg)
    RealT = promote_type(real(dg), ValueT)
    thread_min = fill(typemax(RealT), Threads.maxthreadid())
    thread_max = fill(typemin(RealT), Threads.maxthreadid())
    thread_finite = zeros(Int, Threads.maxthreadid())
    thread_nonfinite = zeros(Int, Threads.maxthreadid())

    @threaded for element in eachelement(dg, cache)
        thread = Threads.threadid()
        value_min = thread_min[thread]
        value_max = thread_max[thread]
        finite_count = thread_finite[thread]
        nonfinite_count = thread_nonfinite[thread]

        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
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

        thread_min[thread] = value_min
        thread_max[thread] = value_max
        thread_finite[thread] = finite_count
        thread_nonfinite[thread] = nonfinite_count
    end

    return reduce_thread_extrema(thread_min, thread_max, thread_finite,
                                 thread_nonfinite)
end
