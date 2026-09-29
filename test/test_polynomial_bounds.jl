@testitem "BoundsPreservingLimiterZhangShu multidimensional transport" setup=[Setup] tags=[
    :misc_part1
] begin
    using LinearAlgebra: kron
    using OrdinaryDiffEqSSPRK: SSPRK33
    using SciMLBase: solve
    using Trixi: BoundsPreservingLimiterZhangShu

    scalar(u, equations) = u[1]
    limiter! = BoundsPreservingLimiterZhangShu(lower = (0.0,), upper = (1.0,),
                                               variables = (scalar,),
                                               polynomial_bounds = true)
    for dimension in (2, 3)
        velocity = ntuple(_ -> 0.1, dimension)
        equations = dimension == 2 ? LinearScalarAdvectionEquation2D(velocity...) :
                    LinearScalarAdvectionEquation3D(velocity...)
        solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
        coordinates_min = ntuple(_ -> -1.0, dimension)
        coordinates_max = ntuple(_ -> 1.0, dimension)
        mesh = TreeMesh(coordinates_min, coordinates_max; initial_refinement_level = 1,
                        periodicity = true)
        initial_condition = (x, t, equations) -> SVector(x[1] < 0 ? 0.0 : 1.0)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                            boundary_conditions = boundary_condition_periodic)
        ode = semidiscretize(semi, (0.0, 0.01))
        mass = Trixi.integrate(ode.u0, semi; normalize = false)
        limiter!(ode.u0, nothing, semi, 0.0)
        sol = solve(ode, SSPRK33(); dt = 1.0e-3, adaptive = false,
                    step_limiter = limiter!, save_everystep = true)
        @test Trixi.SciMLBase.successful_retcode(sol.retcode)
        @test sol.u[end] != sol.u[1]
        @test maximum(sol.u[end]) - minimum(sol.u[end]) > 0.5
        @test all(u -> isapprox(Trixi.integrate(u, semi; normalize = false), mass;
                                rtol = 1.0e-12, atol = 1.0e-12), sol.u)
        interpolation = Trixi.polynomial_interpolation_matrix(solver.basis.nodes,
                                                              range(-1.0, 1.0; length = 11))
        interpolation_nd = reduce(kron, ntuple(_ -> interpolation, dimension))
        profiles = [interpolation_nd * reshape(u, Trixi.nnodes(solver)^dimension, :)
                    for u in sol.u]
        @test all(u -> minimum(u) >= -1.0e-12, profiles)
        @test all(u -> maximum(u) <= 1.0 + 1.0e-12, profiles)

        # Smooth positive transport retains spatial convergence, with the limiter
        # inactive. Compare against an independent exact solution and an unlimited run.
        exact_solution = (x, t, equations) -> SVector(0.5 +
                                                      0.1 * prod(cospi, x .- t .* velocity))
        errors = Float64[]
        for refinement in (2, 3)
            smooth_mesh = TreeMesh(coordinates_min, coordinates_max;
                                   initial_refinement_level = refinement,
                                   periodicity = true)
            smooth_semi = SemidiscretizationHyperbolic(smooth_mesh, equations,
                                                       exact_solution, solver;
                                                       boundary_conditions = boundary_condition_periodic)
            smooth_ode = semidiscretize(smooth_semi, (0.0, 0.001))
            limited = solve(smooth_ode, SSPRK33(); dt = 1.0e-4, adaptive = false,
                            step_limiter = limiter!, save_everystep = false)
            unlimited = solve(smooth_ode, SSPRK33(); dt = 1.0e-4, adaptive = false,
                              save_everystep = false)
            @test Trixi.SciMLBase.successful_retcode(limited.retcode)
            @test Trixi.SciMLBase.successful_retcode(unlimited.retcode)
            @test limited.u[end] == unlimited.u[end]
            exact = Trixi.compute_coefficients(0.001, smooth_semi)
            push!(errors, maximum(abs, limited.u[end] - exact))
        end
        @test errors[2] < errors[1] / 4
    end
end
