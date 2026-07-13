# The MPI example tests live in the `test_mpi_*.jl` files and are tagged `:mpi`.
# They are discovered automatically by `@run_package_tests` and run inside the
# `mpiexec`-launched worker process set up in `runtests.jl`. Items that fail
# often on Windows CI are additionally tagged `:mpi_skip_windows`.

@testitem "MPI supporting functionality" setup=[Setup] tags=[:mpi] begin
    using Trixi: Trixi, ode_norm, SVector
    t = 0.5
    let u = 1.0
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
    let u = [1.0, -2.0]
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
    let u = [SVector(1.0, -2.0), SVector(0.5, -0.1)]
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
end

@testitem "MPI VariableBoundsCallback reductions" setup=[Setup] tags=[:mpi] begin
    using OrdinaryDiffEqLowStorageRK
    using Trixi: Trixi, DGSEM, LinearScalarAdvectionEquation2D,
                 SemidiscretizationHyperbolic, SVector, TreeMesh, VariableBound,
                 VariableBoundsCallback,
                 boundary_condition_periodic, isviolated, mpi_isroot,
                 ode_default_options, semidiscretize

    equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
    initial_condition = (x, t, equations) -> SVector(0.0)
    mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0);
                    initial_refinement_level = 3,
                    periodicity = true)
    solver = DGSEM(polydeg = 1)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_condition_periodic)
    u_ode = Trixi.compute_coefficients(0.0, semi)

    rank_value = if mpi_isroot()
        -0.1
    elseif Trixi.mpi_rank() == Trixi.mpi_nranks() - 1
        1.2
    else
        0.5
    end
    fill!(u_ode, rank_value)

    bound = VariableBound(:scalar, (u, equations) -> u[1]; lower = 0.0, upper = 1.0)
    result = Trixi.evaluate_variable_bounds(u_ode, semi, (bound,)).scalar
    @test result.minimum ≈ -0.1
    @test result.maximum ≈ 1.2
    @test result.lower_violation ≈ 0.1
    @test result.upper_violation ≈ 0.2
    @test result.lower_violated
    @test result.upper_violated

    termination_bound = VariableBound(:impossible, (u, equations) -> u[1]; lower = 1.0)
    termination_callback = VariableBoundsCallback(semi;
                                                  bounds = (termination_bound,),
                                                  interval = 1,
                                                  check_initial = false,
                                                  action = :terminate)
    ode = semidiscretize(semi, (0.0, 0.01))
    sol = solve(ode, RDPK3SpFSAL35();
                dt = 1.0e-3, adaptive = false,
                ode_default_options()..., callback = termination_callback)
    @test sol.t[end] < last(sol.prob.tspan)
    @test termination_callback.affect!.violations_detected == 1
    @test isviolated(termination_callback.affect!.last_results.impossible)
end
