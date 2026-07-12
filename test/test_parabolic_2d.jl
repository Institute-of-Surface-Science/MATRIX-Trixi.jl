@testsnippet Parabolic2D begin
    EXAMPLES_DIR = examples_dir()
end

@testitem "Parabolic2D: TreeMesh componentwise diffusion RHS" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    equations = CompressibleEulerEquations2D(1.4)
    equations_parabolic = LaplaceDiffusionComponentwise2D((0.1, 0.0, 0.0, 0.0),
                                                          equations)
    solver = DGSEM(polydeg = 2)
    mesh = TreeMesh((0.0, 0.0), (1.0, 1.0), initial_refinement_level = 1,
                    n_cells_max = 100, periodicity = true)
    initial_condition = function (x, t, equations)
        rho = 1.0 + 0.1 * sinpi(2 * x[1]) * sinpi(2 * x[2])
        return SVector(rho, 0.0, 0.0, 1.0 / (equations.gamma - 1))
    end

    semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 solver_parabolic = ParabolicFormulationLocalDG(),
                                                 boundary_conditions = (boundary_condition_periodic,
                                                                        boundary_condition_periodic))
    ode = semidiscretize(semi, (0.0, 0.01))
    du = similar(ode.u0)
    @test_nowarn Trixi.rhs_parabolic!(du, ode.u0, semi, 0.0)

    du_wrapped = Trixi.wrap_array(du, semi)
    @test maximum(abs, selectdim(du_wrapped, 1, 1)) > 1.0e-6
    for variable in 2:nvariables(equations)
        @test iszero(maximum(abs, selectdim(du_wrapped, 1, variable)))
    end
end

@testitem "Parabolic2D: DGMulti 2D rhs_parabolic!" setup=[Setup, Parabolic2D] tags=[:parabolic_part1] begin
    using Trixi

    struct SpaceTimeDiffusion2D{NVARS} <:
           Trixi.AbstractEquationsParabolic{2, NVARS,
                                            GradientVariablesConservative} end

    struct ZeroSpeedEquation2D <: Trixi.AbstractEquations{2, 1} end
    @inline Trixi.max_abs_speeds(u, ::ZeroSpeedEquation2D) = SVector(zero(eltype(u)),
                                                                     zero(eltype(u)))

    struct QuadraturePeakDiffusion2D <:
           Trixi.AbstractEquationsParabolic{2, 1,
                                            GradientVariablesConservative} end
    struct UniformNonconstantDiffusion2D{T} <:
           Trixi.AbstractEquationsParabolic{2, 1,
                                            GradientVariablesConservative}
        diffusivity::T
    end

    Trixi.varnames(::typeof(cons2cons), ::SpaceTimeDiffusion2D) = ("scalar",)
    Trixi.varnames(::typeof(cons2prim), ::SpaceTimeDiffusion2D) = ("scalar",)
    Trixi.varnames(::typeof(cons2entropy), ::SpaceTimeDiffusion2D) = ("scalar",)
    @inline Trixi.cons2prim(u, ::SpaceTimeDiffusion2D) = u
    @inline Trixi.cons2entropy(u, ::SpaceTimeDiffusion2D) = u

    @inline function Trixi.flux(u, gradients, orientation::Integer, x, t,
                                ::SpaceTimeDiffusion2D)
        coefficient = 1 + x[1] + 2 * x[2] + t
        return coefficient * gradients[orientation]
    end
    Trixi.have_space_time_dependent_flux(::SpaceTimeDiffusion2D) = Trixi.True()
    Trixi.have_constant_diffusivity(::SpaceTimeDiffusion2D) = Trixi.False()
    @inline Trixi.max_diffusivity(u, x, t, ::SpaceTimeDiffusion2D) = 1 + x[1] +
                                                                     2 * x[2] + t
    Trixi.have_constant_diffusivity(::QuadraturePeakDiffusion2D) = Trixi.False()
    @inline Trixi.max_diffusivity(u, x, t, ::QuadraturePeakDiffusion2D) = 1 +
                                                                          100 *
                                                                          sinpi(2 *
                                                                                x[1])^2 *
                                                                          sinpi(2 *
                                                                                x[2])^2
    Trixi.have_constant_diffusivity(::UniformNonconstantDiffusion2D) = Trixi.False()
    @inline Trixi.max_diffusivity(u, x, t,
    equations::UniformNonconstantDiffusion2D) = equations.diffusivity

    dg = DGMulti(polydeg = 2, element_type = Quad(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central),
                 volume_integral = VolumeIntegralWeakForm())
    cells_per_dimension = (2, 2)
    mesh = DGMultiMesh(dg, cells_per_dimension)

    # test with polynomial initial condition x^2 * y
    # test if we recover the exact second derivative
    initial_condition = (x, t, equations) -> SVector(x[1]^2 * x[2])

    equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
    equations_parabolic = LaplaceDiffusion2D(1.0, equations)

    semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition, dg;
                                                 boundary_conditions = (boundary_condition_periodic,
                                                                        boundary_condition_periodic))
    @trixi_test_nowarn show(stdout, semi)
    @trixi_test_nowarn show(stdout, MIME"text/plain"(), semi)
    @trixi_test_nowarn show(stdout, boundary_condition_do_nothing)

    @test nvariables(semi) == nvariables(equations)
    @test Base.ndims(semi) == Base.ndims(mesh)
    @test Base.real(semi) == Base.real(dg)

    ode = semidiscretize(semi, (0.0, 0.01))
    @test_throws ArgumentError StepsizeCallback(cfl = 0.5,
                                                cfl_parabolic = -0.05)(ode)
    u0 = similar(ode.u0)
    Trixi.compute_coefficients!(u0, 0.0, semi)
    @test u0 ≈ ode.u0

    # test "do nothing" BC just returns first argument
    @test boundary_condition_do_nothing(u0, nothing) == u0

    (; cache, cache_parabolic, equations_parabolic) = semi
    (; gradients) = cache_parabolic
    for dim in eachindex(gradients)
        fill!(gradients[dim], zero(eltype(gradients[dim])))
    end

    # unpack VectorOfArray
    u0 = Base.parent(ode.u0)
    t = 0.0
    # pass in `boundary_condition_periodic` to skip boundary flux/integral evaluation
    parabolic_scheme = semi.solver_parabolic
    Trixi.calc_gradient!(gradients, u0, t, mesh, equations_parabolic,
                         boundary_condition_periodic, dg, parabolic_scheme,
                         cache, cache_parabolic)
    (; x, y, xq, yq) = mesh.md
    @test getindex.(gradients[1], 1) ≈ 2 * xq .* yq
    @test getindex.(gradients[2], 1) ≈ xq .^ 2

    u_flux = similar.(gradients)
    Trixi.calc_parabolic_fluxes!(u_flux, u0, gradients, t, mesh,
                                 have_space_time_dependent_flux(equations_parabolic),
                                 equations_parabolic,
                                 dg, cache, cache_parabolic)
    @test u_flux[1] ≈ gradients[1]
    @test u_flux[2] ≈ gradients[2]

    du = similar(u0)
    Trixi.calc_divergence!(du, u0, t, u_flux, mesh,
                           equations_parabolic,
                           boundary_condition_periodic,
                           dg, semi.solver_parabolic, cache, cache_parabolic)
    Trixi.invert_jacobian!(du, mesh, equations_parabolic, dg, cache; scaling = 1.0)
    @test getindex.(du, 1) ≈ 2 * y

    equations_space_time = SpaceTimeDiffusion2D{1}()
    flux_time = 0.3
    Trixi.calc_parabolic_fluxes!(u_flux, u0, gradients, flux_time, mesh,
                                 have_space_time_dependent_flux(equations_space_time),
                                 equations_space_time,
                                 dg, cache, cache_parabolic)
    coefficient = @. 1 + xq + 2 * yq + flux_time
    @test getindex.(u_flux[1], 1) ≈ coefficient .* getindex.(gradients[1], 1)
    @test getindex.(u_flux[2], 1) ≈ coefficient .* getindex.(gradients[2], 1)

    equations_stepsize = ZeroSpeedEquation2D()
    dt_initial = Trixi.max_dt(u0, 0.0, mesh,
                              have_constant_diffusivity(equations_space_time),
                              equations_stepsize, equations_space_time, dg, cache)
    dt_final = Trixi.max_dt(u0, flux_time, mesh,
                            have_constant_diffusivity(equations_space_time),
                            equations_stepsize, equations_space_time, dg, cache)
    @test dt_final < dt_initial

    semi_parabolic = SemidiscretizationParabolic(mesh, equations_space_time,
                                                 initial_condition, dg;
                                                 boundary_conditions = boundary_condition_periodic)
    ode_parabolic = semidiscretize(semi_parabolic, (0.0, 0.01))
    @test_throws ArgumentError StepsizeCallback()(ode_parabolic)
    @test_throws ArgumentError StepsizeCallback(cfl_parabolic = -0.05)(ode_parabolic)
    stepsize_callback = StepsizeCallback(cfl_parabolic = 0.05)
    dt_parabolic = stepsize_callback(ode_parabolic)
    @test isfinite(dt_parabolic)
    @test dt_parabolic > 0
    @test Trixi.calculate_dt(ode_parabolic.u0, first(ode_parabolic.tspan),
                             1.0, 0.05, semi_parabolic) == dt_parabolic

    equations_constant = LinearDiffusionEquation2D(0.1)
    semi_constant = SemidiscretizationParabolic(mesh, equations_constant,
                                                initial_condition, dg;
                                                boundary_conditions = boundary_condition_periodic)
    ode_constant = semidiscretize(semi_constant, (0.0, 0.01))
    dt_constant = stepsize_callback(ode_constant)
    @test isfinite(dt_constant)
    @test dt_constant > 0

    equations_peak = QuadraturePeakDiffusion2D()
    diffusivity_solution_max = maximum(Trixi.max_diffusivity(u0[i, element],
                                                             SVector(getindex.(mesh.md.xyz,
                                                                               i,
                                                                               element)),
                                                             0.0, equations_peak)
                                       for element in Trixi.eachelement(mesh, dg, cache),
                                           i in Base.OneTo(dg.basis.Np))
    diffusivity_quadrature_max = maximum(Trixi.max_diffusivity(u0[1, element],
                                                               SVector(getindex.(mesh.md.xyzq,
                                                                                 i,
                                                                                 element)),
                                                               0.0, equations_peak)
                                         for element in Trixi.eachelement(mesh, dg,
                                                                          cache),
                                             i in Base.OneTo(dg.basis.Nq))
    @test diffusivity_quadrature_max > diffusivity_solution_max

    equations_uniform = UniformNonconstantDiffusion2D(diffusivity_quadrature_max)
    dt_peak = Trixi.max_dt(u0, 0.0, mesh, Trixi.False(), equations_stepsize,
                           equations_peak, dg, cache)
    dt_uniform = Trixi.max_dt(u0, 0.0, mesh, Trixi.False(), equations_stepsize,
                              equations_uniform, dg, cache)
    @test dt_peak ≈ dt_uniform
end

@testitem "Parabolic2D: Spatially varying diffusivity provider" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    coefficient_function = (x, t, equations) -> one(eltype(x)) +
                                                convert(eltype(x), 0.25) *
                                                sinpi(x[1]) * sinpi(x[2])
    coefficient = SpatiallyVaryingDiffusivity(coefficient_function, 1.25)
    equations = LinearDiffusionEquation2D(coefficient)

    @test Trixi.diffusivity_upper_bound(coefficient) == 1.25
    @test have_constant_diffusivity(equations) == Trixi.False()
    @test have_space_time_dependent_flux(equations) == Trixi.True()
    @test max_diffusivity(SVector(1.0), SVector(0.0, 0.0), 0.0, equations) == 1.25
    @test_throws ArgumentError SpatiallyVaryingDiffusivity(coefficient_function, 0.0)
    @test_throws ArgumentError SpatiallyVaryingDiffusivity(coefficient_function, Inf)
    @test_throws ArgumentError SpatiallyVaryingDiffusivity(coefficient_function, NaN)

    x_left = SVector(-0.5, 0.5)
    x_right = SVector(0.5, 0.5)
    gradients = (SVector(2.0), SVector(-3.0))
    coefficient_left = coefficient_function(x_left, 0.0, equations)
    coefficient_right = coefficient_function(x_right, 0.0, equations)
    @test coefficient_left == 0.75
    @test coefficient_right == 1.25
    for orientation in 1:2
        flux_left = flux(SVector(1.0), gradients, orientation, x_left, 0.0,
                         equations)
        flux_right = flux(SVector(1.0), gradients, orientation, x_right, 0.0,
                          equations)
        @test flux_right[1] / flux_left[1] ≈ coefficient_right / coefficient_left
    end

    time_coefficient = SpatiallyVaryingDiffusivity((x, t, equations) -> one(eltype(x)) +
                                                                        t,
                                                   2.0)
    time_equations = LinearDiffusionEquation2D(time_coefficient)
    @test flux(SVector(1.0), gradients, 1, x_left, 0.5,
               time_equations) ≈ 1.5 * gradients[1]

    equations_hyperbolic = LinearScalarAdvectionEquation2D(1.0, 1.0)
    equations_laplace = LaplaceDiffusion2D(coefficient, equations_hyperbolic)
    @test have_constant_diffusivity(equations_laplace) == Trixi.False()
    @test have_space_time_dependent_flux(equations_laplace) == Trixi.True()
    @test flux(SVector(1.0), gradients, 1, x_right, 0.0,
               equations_laplace) ≈ SVector(coefficient_right * gradients[1])

    adapted = Trixi.trixi_adapt(Array, Float32, equations)
    @test adapted.diffusivity isa SpatiallyVaryingDiffusivity{<:Any, Float32}
    @test adapted.diffusivity.value_function === coefficient_function
    @test adapted.diffusivity.upper_bound == 1.25f0
    @test Trixi.diffusivity_value(adapted.diffusivity,
                                  SVector(0.5f0, 0.5f0), 0.0f0,
                                  adapted) isa Float32

    solver = DGSEM(polydeg = 3)
    mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0);
                    initial_refinement_level = 1,
                    n_cells_max = 100,
                    periodicity = true)
    initial_condition = (x, t, equations) -> SVector(sinpi(x[1]) * sinpi(x[2]))
    equations_constant = LinearDiffusionEquation2D(0.1)
    @test equations_constant.diffusivity isa ConstantDiffusivity{Float64}
    @test have_constant_diffusivity(equations_constant) == Trixi.True()
    coefficient_constant = SpatiallyVaryingDiffusivity((x, t, equations) -> 0.1,
                                                       0.1)
    equations_variable_constant = LinearDiffusionEquation2D(coefficient_constant)

    semi_constant = SemidiscretizationParabolic(mesh, equations_constant,
                                                initial_condition, solver;
                                                solver_parabolic = ParabolicFormulationBassiRebay1(),
                                                boundary_conditions = boundary_condition_periodic)
    semi_variable_constant = SemidiscretizationParabolic(mesh,
                                                         equations_variable_constant,
                                                         initial_condition, solver;
                                                         solver_parabolic = ParabolicFormulationBassiRebay1(),
                                                         boundary_conditions = boundary_condition_periodic)
    ode_constant = semidiscretize(semi_constant, (0.0, 0.01))
    ode_variable_constant = semidiscretize(semi_variable_constant, (0.0, 0.01))
    du_constant = similar(ode_constant.u0)
    du_variable_constant = similar(ode_variable_constant.u0)
    Trixi.rhs_parabolic!(du_constant, ode_constant.u0, semi_constant, 0.0)
    Trixi.rhs_parabolic!(du_variable_constant, ode_variable_constant.u0,
                         semi_variable_constant, 0.0)

    @test du_variable_constant≈du_constant atol=10 * eps() rtol=10 * eps()
    stepsize_callback = StepsizeCallback(cfl_parabolic = 0.05)
    @test stepsize_callback(ode_variable_constant) ≈ stepsize_callback(ode_constant)
    @test_throws ArgumentError linear_structure(semi_variable_constant)
end

@testitem "Parabolic2D: DGMulti LDG diffusion core" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    using OrdinaryDiffEqLowStorageRK

    struct VariablePenaltyDiffusion2D{F} <: Trixi.AbstractLaplaceDiffusion{2, 1}
        diffusivity::F
    end

    Trixi.varnames(::typeof(cons2cons), ::VariablePenaltyDiffusion2D) = ("scalar",)
    Trixi.varnames(::typeof(cons2prim), ::VariablePenaltyDiffusion2D) = ("scalar",)
    Trixi.varnames(::typeof(cons2entropy), ::VariablePenaltyDiffusion2D) = ("scalar",)
    Trixi.cons2prim(u, ::VariablePenaltyDiffusion2D) = u
    Trixi.cons2entropy(u, ::VariablePenaltyDiffusion2D) = u
    Trixi.have_constant_diffusivity(::VariablePenaltyDiffusion2D) = Trixi.False()
    Trixi.have_space_time_dependent_flux(::VariablePenaltyDiffusion2D) = Trixi.True()
    @inline function Trixi.max_diffusivity(u, x, t,
                                           equations::VariablePenaltyDiffusion2D)
        return equations.diffusivity(x, t)
    end
    @inline function Trixi.flux(u, gradients, orientation::Integer, x, t,
                                equations::VariablePenaltyDiffusion2D)
        return equations.diffusivity(x, t) * gradients[orientation]
    end

    dg = DGMulti(polydeg = 2, element_type = Quad(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central),
                 volume_integral = VolumeIntegralWeakForm())
    equations = LinearDiffusionEquation2D(0.1)
    initial_condition = (x, t, equations) -> SVector(sinpi(x[1]) * sinpi(x[2]))

    periodic_mesh = DGMultiMesh(dg, (3, 3), periodicity = true)
    semi = SemidiscretizationParabolic(periodic_mesh, equations, initial_condition, dg;
                                       solver_parabolic = ParabolicFormulationLocalDG(1.0),
                                       boundary_conditions = boundary_condition_periodic)
    ode = semidiscretize(semi, (0.0, 0.01))
    sol = solve(ode, RDPK3SpFSAL35(); abstol = 1.0e-9, reltol = 1.0e-9,
                save_everystep = false)
    @test Trixi.SciMLBase.successful_retcode(sol.retcode)
    @test sum(abs2, sol.u[end]) < sum(abs2, sol.u[1])

    # Diagonal triangular faces have equal-magnitude normal components. The LDG
    # switch must still be opposite on both sides so periodic diffusion conserves mass.
    dg_tri = DGMulti(polydeg = 2, element_type = Tri(),
                     approximation_type = Polynomial(),
                     surface_integral = SurfaceIntegralWeakForm(flux_central),
                     volume_integral = VolumeIntegralWeakForm())
    periodic_mesh_tri = DGMultiMesh(dg_tri, (2, 2), periodicity = true)
    semi_tri = SemidiscretizationParabolic(periodic_mesh_tri, equations,
                                           initial_condition, dg_tri;
                                           solver_parabolic = ParabolicFormulationLocalDG(1.0),
                                           boundary_conditions = boundary_condition_periodic)
    ode_tri = semidiscretize(semi_tri, (0.0, 0.01))
    u_tri = copy(ode_tri.u0)
    du_tri = similar(u_tri)
    Trixi.rhs_parabolic!(du_tri, u_tri, semi_tri, 0.0)
    integrated_rhs_tri = Trixi.integrate(du_tri, semi_tri; normalize = false)
    @test maximum(abs, integrated_rhs_tri) < 1.0e-12

    constant_initial_condition = (x, t, equations) -> SVector(one(x[1]))
    for parabolic_scheme in (ParabolicFormulationLocalDG(),
                             ParabolicFormulationLocalDG(0.0))
        semi_periodic = SemidiscretizationParabolic(periodic_mesh, equations,
                                                    constant_initial_condition, dg;
                                                    solver_parabolic = parabolic_scheme,
                                                    boundary_conditions = boundary_condition_periodic)
        ode_periodic = semidiscretize(semi_periodic, (0.0, 0.01))
        du_periodic = similar(ode_periodic.u0)
        @test_nowarn Trixi.rhs_parabolic!(du_periodic, ode_periodic.u0,
                                          semi_periodic, 0.0)
        @test maximum(abs, du_periodic) < 1.0e-11
    end

    penalty_parameter = 2.0
    semi_stronger = remake(semi;
                           solver_parabolic = ParabolicFormulationLocalDG(10 *
                                                                          penalty_parameter))
    semi_penalty = remake(semi;
                          solver_parabolic = ParabolicFormulationLocalDG(penalty_parameter))
    ode_stronger = semidiscretize(semi_stronger, (0.0, 0.01))
    ode_penalty = semidiscretize(semi_penalty, (0.0, 0.01))
    stepsize_callback = StepsizeCallback(cfl_parabolic = 1.0)
    @test stepsize_callback(ode_stronger) < stepsize_callback(ode_penalty)

    left(x, tol = 50 * eps()) = abs(x[1] + 1) < tol
    right(x, tol = 50 * eps()) = abs(x[1] - 1) < tol
    bottom(x, tol = 50 * eps()) = abs(x[2] + 1) < tol
    top(x, tol = 50 * eps()) = abs(x[2] - 1) < tol
    is_on_boundary = (; left, right, bottom, top)
    physical_mesh = DGMultiMesh(dg, (2, 2); is_on_boundary)
    boundary_value = (x, t, equations) -> initial_condition(x, t, equations)
    dirichlet = BoundaryConditionDirichlet(boundary_value)
    boundary_conditions = (; left = dirichlet, right = dirichlet,
                           bottom = dirichlet, top = dirichlet)

    variable_equations = VariablePenaltyDiffusion2D((x, t) -> 0.1 *
                                                              (1 + x[1]^2 + t))
    semi_variable = SemidiscretizationParabolic(physical_mesh, variable_equations,
                                                initial_condition, dg;
                                                solver_parabolic = ParabolicFormulationLocalDG(1.0),
                                                boundary_conditions)
    ode_variable = semidiscretize(semi_variable, (0.0, 0.01))
    du_variable = similar(ode_variable.u0)
    @test_nowarn Trixi.rhs_parabolic!(du_variable, ode_variable.u0,
                                      semi_variable, 0.2)
    @test all(isfinite, du_variable)

    for parabolic_scheme in (ParabolicFormulationLocalDG(),
                             ParabolicFormulationLocalDG(0.0),
                             ParabolicFormulationLocalDG(-1.0))
        @test_throws ArgumentError SemidiscretizationParabolic(physical_mesh, equations,
                                                               initial_condition, dg;
                                                               solver_parabolic = parabolic_scheme,
                                                               boundary_conditions)
    end

    affine_initial_condition = (x, t, equations) -> SVector(x[1] + x[2])
    normal_flux_negative = (x, t, equations) -> SVector(-max_diffusivity(equations))
    normal_flux_positive = (x, t, equations) -> SVector(max_diffusivity(equations))
    boundary_conditions_neumann = (;
                                   left = BoundaryConditionNeumann(normal_flux_negative),
                                   right = BoundaryConditionNeumann(normal_flux_positive),
                                   bottom = BoundaryConditionNeumann(normal_flux_negative),
                                   top = BoundaryConditionNeumann(normal_flux_positive))
    for parabolic_scheme in (ParabolicFormulationBassiRebay1(),
                             ParabolicFormulationLocalDG(1.0))
        semi_neumann = SemidiscretizationParabolic(physical_mesh, equations,
                                                   affine_initial_condition, dg;
                                                   solver_parabolic = parabolic_scheme,
                                                   boundary_conditions = boundary_conditions_neumann)
        ode_neumann = semidiscretize(semi_neumann, (0.0, 0.01))
        du_neumann = similar(ode_neumann.u0)
        Trixi.rhs_parabolic!(du_neumann, ode_neumann.u0, semi_neumann, 0.0)
        @test maximum(abs, du_neumann) < 1.0e-11
    end

    u_inner = SVector(1.0)
    u_outer = SVector(2.0)
    @test Trixi.penalty(u_outer, u_inner, 3.0, equations,
                        ParabolicFormulationLocalDG(2.0)) ≈ SVector(0.6)
    x_penalty = SVector(0.5, 0.0)
    t_penalty = 0.2
    expected_diffusivity = variable_equations.diffusivity(x_penalty, t_penalty)
    @test Trixi.penalty(u_outer, u_inner, 3.0, x_penalty, t_penalty,
                        variable_equations,
                        ParabolicFormulationLocalDG(2.0)) ≈
          SVector(6 * expected_diffusivity)

    carrier = CompressibleEulerEquations2D(1.4)
    equations_componentwise = LaplaceDiffusionComponentwise2D((0.1, 0.0, 0.0, 0.0),
                                                              carrier)
    jump_inner = SVector(1.0, 2.0, 3.0, 4.0)
    jump_outer = SVector(2.0, 4.0, 6.0, 8.0)
    componentwise_penalty = Trixi.penalty(jump_outer, jump_inner, 3.0,
                                          equations_componentwise,
                                          ParabolicFormulationLocalDG(2.0))
    @test componentwise_penalty[1] ≈ 0.6
    @test all(iszero, componentwise_penalty[2:end])
end

@testitem "Parabolic2D: Space- and time-dependent parabolic flux coordinates" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    using Trixi

    struct SpaceTimeDiffusion2D{NVARS} <:
           Trixi.AbstractEquationsParabolic{2, NVARS,
                                            GradientVariablesConservative} end

    @inline function Trixi.flux(u, gradients, orientation::Integer, x, t,
                                ::SpaceTimeDiffusion2D)
        coefficient = 1 + x[1] + 2 * x[2] + t
        return coefficient * gradients[orientation]
    end
    Trixi.have_space_time_dependent_flux(::SpaceTimeDiffusion2D) = Trixi.True()
    Trixi.have_constant_diffusivity(::SpaceTimeDiffusion2D) = Trixi.False()
    @inline Trixi.max_diffusivity(u, x, t, ::SpaceTimeDiffusion2D) = 1 + x[1] +
                                                                     2 * x[2] + t

    function test_space_time_parabolic_flux_coordinates(mesh)
        equations = LinearScalarAdvectionEquation2D(0.0, 0.0)
        equations_parabolic = LaplaceDiffusion2D(1.0, equations)
        solver = DGSEM(polydeg = 2)
        initial_condition = (x, t, equations) -> SVector(sinpi(2 * x[1]) *
                                                         sinpi(2 * x[2]))
        semi = SemidiscretizationHyperbolicParabolic(mesh,
                                                     (equations, equations_parabolic),
                                                     initial_condition, solver;
                                                     boundary_conditions = (boundary_condition_periodic,
                                                                            boundary_condition_periodic))
        ode = semidiscretize(semi, (0.0, 0.01))
        du = similar(ode.u0)
        Trixi.rhs_parabolic!(du, ode.u0, semi, 0.0)

        (; u_transformed, gradients, flux_parabolic) = semi.cache_parabolic.parabolic_container
        equations_space_time = SpaceTimeDiffusion2D{1}()
        flux_time = 0.3
        Trixi.calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed,
                                     flux_time, mesh,
                                     have_space_time_dependent_flux(equations_space_time),
                                     equations_space_time, solver, semi.cache)

        node_coordinates = semi.cache.elements.node_coordinates
        for element in Trixi.eachelement(solver, semi.cache),
            j in Trixi.eachnode(solver), i in Trixi.eachnode(solver)

            x_node = Trixi.get_node_coords(node_coordinates, equations_space_time,
                                           solver, i, j, element)
            coefficient = 1 + x_node[1] + 2 * x_node[2] + flux_time
            for orientation in 1:2
                gradient_node = Trixi.get_node_vars(gradients[orientation],
                                                    equations_space_time, solver,
                                                    i, j, element)
                flux_node = Trixi.get_node_vars(flux_parabolic[orientation],
                                                equations_space_time, solver,
                                                i, j, element)
                @test flux_node ≈ coefficient * gradient_node
            end
        end

        u = Trixi.wrap_array(ode.u0, semi)
        dt_initial = Trixi.max_dt(u, 0.0, mesh,
                                  have_constant_diffusivity(equations_space_time),
                                  equations, equations_space_time, solver, semi.cache)
        dt_final = Trixi.max_dt(u, flux_time, mesh,
                                have_constant_diffusivity(equations_space_time),
                                equations, equations_space_time, solver, semi.cache)
        @test dt_final < dt_initial
    end

    tree_mesh = TreeMesh((0.0, 0.0), (1.0, 1.0),
                         initial_refinement_level = 1,
                         n_cells_max = 100,
                         periodicity = true)
    test_space_time_parabolic_flux_coordinates(tree_mesh)

    p4est_mesh = P4estMesh((2, 2), polydeg = 2,
                           coordinates_min = (0.0, 0.0),
                           coordinates_max = (1.0, 1.0),
                           periodicity = true)
    test_space_time_parabolic_flux_coordinates(p4est_mesh)
end

@testitem "Parabolic2D: DGMulti tagged triangular diffusion" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_diffusion_triangulate_pkg_mesh.jl"),
                        polydeg=2, mesh_size=0.45, tspan=(0.0, 0.05),
                        l2=[0.000866366165081298],
                        linf=[0.0053946772230686335])
    @test Trixi.SciMLBase.successful_retcode(sol.retcode)
    coarse_l2_error, coarse_linf_error = analysis_callback(sol)
    @test all([0.00023559213035217064] .< coarse_l2_error)
    @test all([0.001535851920118958] .< coarse_linf_error)
    @test semi.solver_parabolic isa ParabolicFormulationLocalDG
    @test semi.solver_parabolic.penalty_parameter > 0

    expected_boundary_names = Set((:bottom, :right, :top, :left))
    @test Set(keys(mesh.boundary_faces)) == expected_boundary_names
    @test all(name -> !isempty(mesh.boundary_faces[name]), expected_boundary_names)

    face_connectivity = vec(mesh.md.FToF)
    all_boundary_faces = findall(face_connectivity .== eachindex(face_connectivity))
    tagged_boundary_faces = sort!(vcat(values(mesh.boundary_faces)...))
    @test tagged_boundary_faces == all_boundary_faces
    @test allunique(tagged_boundary_faces)

    function boundary_coordinate_values(mesh, solver, boundary_name,
                                        coordinate_dimension)
        number_of_faces = StartUpDG.num_faces(solver.basis.element_type)
        points_per_face = solver.basis.Nfq ÷ number_of_faces
        coordinate_values = Float64[]
        for face_id in mesh.boundary_faces[boundary_name]
            element_id = (face_id - 1) ÷ number_of_faces + 1
            local_face = (face_id - 1) % number_of_faces
            face_nodes = (local_face * points_per_face + 1):((local_face + 1) * points_per_face)
            append!(coordinate_values,
                    mesh.md.xyzf[coordinate_dimension][face_nodes, element_id])
        end
        return coordinate_values
    end

    coordinate_tolerance = 500 * eps(Float64)
    @test all(value -> isapprox(value, -1.0; atol = coordinate_tolerance, rtol = 0.0),
              boundary_coordinate_values(mesh, solver, :bottom, 2))
    @test all(value -> isapprox(value, 1.0; atol = coordinate_tolerance, rtol = 0.0),
              boundary_coordinate_values(mesh, solver, :right, 1))
    @test all(value -> isapprox(value, 1.0; atol = coordinate_tolerance, rtol = 0.0),
              boundary_coordinate_values(mesh, solver, :top, 2))
    @test all(value -> isapprox(value, -1.0; atol = coordinate_tolerance, rtol = 0.0),
              boundary_coordinate_values(mesh, solver, :left, 1))

    steady_initial_condition = (x, t, equations) -> SVector(one(x[2]) + x[2])
    semi_steady = SemidiscretizationParabolic(mesh, equations,
                                              steady_initial_condition, solver;
                                              solver_parabolic,
                                              boundary_conditions)
    ode_steady = semidiscretize(semi_steady, tspan)
    du_steady = similar(ode_steady.u0)
    Trixi.rhs_parabolic!(du_steady, ode_steady.u0, semi_steady, first(tspan))
    @test maximum(abs, du_steady) < 1.0e-11

    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: DGMulti tagged triangular diffusion refinement" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_diffusion_triangulate_pkg_mesh.jl"),
                        polydeg=2, mesh_size=0.3, tspan=(0.0, 0.05),
                        l2=[0.00023559213035217064],
                        linf=[0.001535851920118958])
    @test Trixi.SciMLBase.successful_retcode(sol.retcode)
    fine_l2_error, fine_linf_error = analysis_callback(sol)
    @test all(fine_l2_error .< [0.000866366165081298])
    @test all(fine_linf_error .< [0.0053946772230686335])
end

@testitem "Parabolic2D: DGMulti: elixir_advection_diffusion.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_advection_diffusion.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.2485803335154642],
                        linf=[1.079606969242132])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: DGMulti: elixir_advection_diffusion_periodic.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_advection_diffusion_periodic.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.03180371984888462],
                        linf=[0.2136821621370909])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: DGMulti: elixir_advection_diffusion_nonperiodic.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.002123168335604323],
                        linf=[0.00963640423513712])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: DGMulti: elixir_navierstokes_convergence.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_navierstokes_convergence.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[
                            0.0015355076237431118,
                            0.003384316785885901,
                            0.0036531858026850757,
                            0.009948436101649498
                        ],
                        linf=[
                            0.005522560543588462,
                            0.013425258431728926,
                            0.013962115936715924,
                            0.027483099961148838
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: DGMulti: elixir_navierstokes_convergence_curved.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_navierstokes_convergence_curved.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[
                            0.0042551020940351444,
                            0.011118489080358264,
                            0.011281831362358863,
                            0.035736565778376306
                        ],
                        linf=[
                            0.015071709836357083,
                            0.04103131887989486,
                            0.03990424032494211,
                            0.13094018584692968
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: DGMulti: elixir_navierstokes_lid_driven_cavity.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.5),
                        l2=[
                            0.0002215612357465129,
                            0.028318325887331217,
                            0.009509168805093485,
                            0.028267893004691534
                        ],
                        linf=[
                            0.0015622793960574644,
                            0.1488665309341318,
                            0.07163235778907852,
                            0.19472797949052278
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.4), polydeg=5,
                        l2=[4.0915532997994255e-6],
                        linf=[2.3040850347877395e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion.jl (Gauss-Legendre)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        solver=DGSEM(polydeg = 5, surface_flux = flux_lax_friedrichs,
                                     basis_type = GaussLegendreBasis),
                        initial_refinement_level=2, tspan=(0.0, 0.4),
                        l2=[2.8254621369070895e-6], linf=[6.914648264633172e-6])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        initial_refinement_level=2, tspan=(0.0, 0.4), polydeg=5,
                        l2=[6.193056910594806e-6], linf=[4.918855889635143e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_gradient_source_terms.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.4), polydeg=3,
                        l2=[0.0015151641634070158], linf=[0.00956001376594895])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_gradient_source_terms.jl (Fixed timestep)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.4),
                        solver_parabolic=ParabolicFormulationBassiRebay1(), nu=1e-3,
                        stepsize_callback=TrivialCallback(), dt=1e-1,
                        l2=[0.0017395186758592685], linf=[0.007481527467476025])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_gradient_source_terms.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        mesh=P4estMesh((4, 4), polydeg = 1,
                                       coordinates_min = coordinates_min,
                                       coordinates_max = coordinates_max,
                                       periodicity = true),
                        tspan=(0.0, 0.4),
                        solver_parabolic=ParabolicFormulationBassiRebay1(), nu=1e-3,
                        stepsize_callback=TrivialCallback(), dt=1e-1,
                        l2=[0.0017395186758592685], linf=[0.007481527467476025])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion.jl (Refined mesh)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        tspan=(0.0, 0.0))
    LLID = Trixi.local_leaf_cells(mesh.tree)
    num_leaves = length(LLID)
    @assert num_leaves % 8 == 0
    Trixi.refine!(mesh.tree, LLID[1:Int(num_leaves / 8)])
    tspan = (0.0, 1.5)
    semi = SemidiscretizationHyperbolicParabolic(mesh,
                                                 (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 boundary_conditions = (boundary_conditions,
                                                                        boundary_conditions_parabolic))
    ode = semidiscretize(semi, tspan)
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
    callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)
    sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
                ode_default_options()..., callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈ [1.67452550744728e-6]
    @test linf_error ≈ [7.905059166368744e-6]

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 100)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 100)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_amr.jl"),
                        initial_refinement_level=2,
                        base_level=2,
                        med_level=3,
                        max_level=4,
                        l2=[0.0009662045510830027],
                        linf=[0.006121646998993091])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_nonperiodic.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        l2=[0.007646800618485118],
                        linf=[0.10067621050468958])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_nonperiodic.jl (Gauss-Legendre)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        solver=DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
                                     basis_type = GaussLegendreBasis),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        l2=[0.005916696880764326],
                        linf=[0.034212013224034776])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        tspan=(0.0, 0.01),
                        l2=[0.0007711488519400885],
                        linf=[0.015254743335726637])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        tspan=(0.0, 0.01),
                        l2=[0.000684755734524055],
                        linf=[0.01141444199847298])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_imex_operator.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_imex_operator.jl"),
                        l2=[7.542670562162156e-8], linf=[3.972014046560446e-7])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    # Compile Trixi.rhs_parabolic! by calling it once before checking
    # allocations
    Trixi.rhs_parabolic!(similar(sol.u[end]), copy(sol.u[end]), semi, sol.t[end])
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_nonperiodic.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        l2=[0.007009146246373517], linf=[0.09535203925012649])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        analysis_callback=AnalysisCallback(semi,
                                                           interval = analysis_interval,
                                                           extra_analysis_integrals = (energy_kinetic,
                                                                                       energy_internal,
                                                                                       enstrophy)),
                        l2=[
                            0.0021116725306624543,
                            0.003432235149083229,
                            0.003874252819605527,
                            0.012469246082535005
                        ],
                        linf=[
                            0.012006418939279007,
                            0.03552087120962882,
                            0.02451274749189282,
                            0.11191122588626357
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl (isothermal walls)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    using Trixi: Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)),
                        l2=[
                            0.0021036296503840883,
                            0.003435843933397192,
                            0.003867359878114748,
                            0.012670355349293195
                        ],
                        linf=[
                            0.01200626179308184,
                            0.03550212518997239,
                            0.025107947320178275,
                            0.11647078036751068
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl (Entropy gradient variables)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            0.002140374251729127,
                            0.003425828709496601,
                            0.0038915122887358097,
                            0.012506862342858291
                        ],
                        linf=[
                            0.012244412004772665,
                            0.03507559186131113,
                            0.02458089234472249,
                            0.11425600758024679
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl (Entropy gradient variables, isothermal walls)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        gradient_variables=GradientVariablesEntropy(),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)),
                        l2=[
                            0.0021349737347923716,
                            0.0034301388278178365,
                            0.0038928324473968836,
                            0.012693611436338
                        ],
                        linf=[
                            0.012244236275761766,
                            0.03505406631430898,
                            0.025099598505644406,
                            0.11795616324985403
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl (flux differencing)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        volume_integral=VolumeIntegralFluxDifferencing(flux_central),
                        l2=[
                            0.0021116725306612075,
                            0.0034322351490838703,
                            0.0038742528196011594,
                            0.012469246082545557
                        ],
                        linf=[
                            0.012006418939262131,
                            0.0355208712096602,
                            0.024512747491999436,
                            0.11191122588669522
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl (Refined mesh)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        tspan=(0.0, 0.0), initial_refinement_level=3)
    LLID = Trixi.local_leaf_cells(mesh.tree)
    num_leaves = length(LLID)
    @assert num_leaves % 4 == 0
    Trixi.refine!(mesh.tree, LLID[1:Int(num_leaves / 4)])
    tspan = (0.0, 0.5)
    semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 boundary_conditions = (boundary_conditions,
                                                                        boundary_conditions_parabolic),
                                                 source_terms = source_terms_navier_stokes_convergence_test)
    ode = semidiscretize(semi, tspan)
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
    callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)
    sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
                dt = 1e-5,
                ode_default_options()..., callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈
          [0.00024296959174050973;
           0.00020932631586399853;
           0.0005390572390981241;
           0.00026753561391316933]
    @test linf_error ≈
          [0.0016210102053486608;
           0.0025932876486537016;
           0.0029539073438284817;
           0.0020771191202548778]
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_lid_driven_cavity.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.5),
                        l2=[
                            0.00015144571529699053,
                            0.018766076072331623,
                            0.007065070765652574,
                            0.0208399005734258
                        ],
                        linf=[
                            0.0014523369373669048,
                            0.12366779944955864,
                            0.05532450997115432,
                            0.16099927805328207
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_shearlayer_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_shearlayer_amr.jl"),
                        l2=[
                            0.0003791330378826807,
                            0.09969310227471417,
                            0.021308426486134697,
                            0.09122052535455805
                        ],
                        linf=[
                            0.001566286906820924,
                            0.6899677733045734,
                            0.0643250936249398,
                            0.5512608650028881
                        ],
                        tspan=(0.0, 0.2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_shearlayer_nonconforming.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_shearlayer_nonconforming.jl"),
                        l2=[
                            0.005392265705768764,
                            0.577092995962308,
                            0.6217677551929198,
                            1.18789128045329
                        ],
                        linf=[
                            0.02761149200829227,
                            1.2597266873423871,
                            1.3269424500005702,
                            6.780348096271325
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_shearlayer_nonconforming.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_shearlayer_nonconforming.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        l2=[
                            0.005352370793583371,
                            0.5969444914287823,
                            0.6317300073130132,
                            1.1794786369127415
                        ],
                        linf=[
                            0.027301485183723107,
                            1.418851037417376,
                            1.3376325628056016,
                            6.713351975092763
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_taylor_green_vortex_sutherland.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_taylor_green_vortex_sutherland.jl"),
                        l2=[
                            0.001452856280034929,
                            0.0007538775539989481,
                            0.0007538775539988681,
                            0.011035506549989587
                        ],
                        linf=[
                            0.003291912841311362,
                            0.002986462478096974,
                            0.0029864624780958637,
                            0.0231954665514138
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_viscous_shock.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        l2=[
                            2.817640352994614e-5,
                            1.3827801939742e-5,
                            3.1001993851549174e-17,
                            1.7535689010948764e-5
                        ],
                        linf=[
                            0.0002185837290411552,
                            0.00013405261969601234,
                            1.8273738729889617e-16,
                            0.00015782934605046428
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_viscous_shock.jl (Gauss-Legendre, LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        solver=DGSEM(polydeg = 3, surface_flux = flux_hlle,
                                     basis_type = GaussLegendreBasis),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        cfl_parabolic=0.04,
                        l2=[
                            6.599006355897759e-6,
                            4.514805201434994e-6,
                            6.54834144833621e-17,
                            4.882545625516753e-6
                        ],
                        linf=[
                            3.7580718253771295e-5,
                            2.6691756676799905e-5,
                            3.560074538214949e-16,
                            2.989434893274634e-5
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_periodic.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.0023754695605828443],
                        linf=[0.008154128363741964])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_rotated.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_rotated.jl"),
                        l2=[4.8533724384822306e-5], linf=[0.0006284491001110615])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_periodic_curved.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic_curved.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.006708147442490916],
                        linf=[0.04807038397976693])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_periodic_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic_amr.jl"),
                        tspan=(0.0, 0.01),
                        l2=[4.49813737871379e-5],
                        linf=[0.0001874702347290924])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: diffusion AMR flux conservation" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_diffusion_amr_flux_conservation.jl"),
                        l2=[0.022994760410297986],
                        linf=[0.10346147528337513])

    @test initial_ncells == 4
    @test final_ncells > initial_ncells
    @test final_nmortars > 0
    @test final_ncells < 500

    @test boundary_fluxes.x_neg≈expected_flux_x_neg atol=1.0e-11
    @test boundary_fluxes.x_pos≈expected_flux_x_pos atol=1.0e-11
    @test boundary_fluxes.y_neg≈expected_flux_y_neg atol=1.0e-11
    @test boundary_fluxes.y_pos≈expected_flux_y_pos atol=1.0e-11
    @test net_boundary_flux≈0.0 atol=1.0e-11

    @test mass_rate≈net_boundary_flux atol=1.0e-10 rtol=1.0e-10
    @test final_mass≈initial_mass atol=1.0e-10 rtol=1.0e-10

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        tspan=(0.0, 0.01),
                        l2=[0.0007711488519390165],
                        linf=[0.015254743335703765])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl (Parabolic CFL)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        initial_refinement_level=2,
                        callbacks=CallbackSet(summary_callback, analysis_callback,
                                              alive_callback,
                                              StepsizeCallback(cfl = 1.6,
                                                               cfl_parabolic = 0.2)),
                        ode_alg=CarpenterKennedy2N54(williamson_condition = false),
                        dt=1.0, # will be overwritten
                        l2=[0.00010850375815619432],
                        linf=[0.0024081141187764932])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        tspan=(0.0, 0.01),
                        l2=[0.0006847533999311489],
                        linf=[0.01141430509080712])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_nonperiodic_curved.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_curved.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.00919917034843865],
                        linf=[0.14186297438393505])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_convergence.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=1, tspan=(0.0, 0.2),
                        l2=[
                            0.0003811978986531135,
                            0.0005874314969137914,
                            0.0009142898787681551,
                            0.0011613918893790497
                        ],
                        linf=[
                            0.0021633623985426453,
                            0.009484348273965089,
                            0.0042315720663082534,
                            0.011661660264076446
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_convergence_nonperiodic.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_convergence_nonperiodic.jl"),
                        initial_refinement_level=1, tspan=(0.0, 0.2),
                        l2=[
                            0.0004036496258545996,
                            0.0005869762480189079,
                            0.0009148853742181908,
                            0.0011984191532764543
                        ],
                        linf=[
                            0.0024993634989209923,
                            0.009487866203496731,
                            0.004505829506103787,
                            0.011634902753554499
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_lid_driven_cavity.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.5),
                        l2=[
                            0.00028716166408816073,
                            0.08101204560401647,
                            0.02099595625377768,
                            0.05008149754143295
                        ],
                        linf=[
                            0.014804500261322406,
                            0.9513271652357098,
                            0.7223919625994717,
                            1.4846907331004786
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_lid_driven_cavity_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity_amr.jl"),
                        tspan=(0.0, 1.0),
                        l2=[
                            0.0005305062668392406, 0.07898766423138423,
                            0.02928291476915143, 0.11715767702870489
                        ],
                        linf=[
                            0.005950675379011616, 0.9254051903170974,
                            0.7991033306362262, 1.694659748457326
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_lid_driven_cavity_amr.jl (IndicatorNodalFunction)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity_amr.jl"),
                        tspan=(0.0, 2.5),
                        amr_indicator=IndicatorNodalFunction((u, x, t) -> ((x[1] <
                                                                            sin(π * t)) &&
                                                                           (x[2] <
                                                                            sin(π * t))) ?
                                                                          1.0 : 0.0,
                                                             semi),
                        l2=[
                            0.000751796085921976,
                            0.10544344448905413,
                            0.05559123730159854,
                            0.13538844542176662
                        ],
                        linf=[
                            0.018687948537448595,
                            0.9693988005334362,
                            0.6314735963971362,
                            1.9961130828380647
                        ],
                        atol=1e-4,
                        rtol=1e-6)
    # Ensure that the mesh size did not change to test IndicatorNodalFunction
    #expected N_ele(t=2.5) = 576, N_ele(t=5) = 303, N_ele(t=7.5) = 51, N_ele(t=10) = 111
    @test nelements(semi.cache.elements) == 576
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_shearlayer_nonconforming.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_shearlayer_nonconforming.jl"),
                        l2=[
                            0.005392265705780167,
                            0.5770929959613263,
                            0.6217677551915209,
                            1.1878912804557846
                        ],
                        linf=[
                            0.027611492008346672,
                            1.2597266873187805,
                            1.3269424500029385,
                            6.7803480962822675
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_shearlayer_nonconforming.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_shearlayer_nonconforming.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        l2=[
                            0.0053523707935916025,
                            0.5969444914278867,
                            0.6317300073116101,
                            1.1794786369145007
                        ],
                        linf=[
                            0.027301485183779173,
                            1.4188510374095429,
                            1.3376325628060792,
                            6.7133519750896085
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_NACA0012airfoil_mach08.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_NACA0012airfoil_mach08.jl"),
                        l2=[0.000186486564226516,
                            0.0005076712323400374,
                            0.00038074588984354107,
                            0.002128177239782089],
                        linf=[0.5153387072802718,
                            1.199362305026636,
                            0.9077214424040279,
                            5.666071182328691], tspan=(0.0, 0.001),
                        initial_refinement_level=0)

    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode) # Just a placeholder in this case

    u = Trixi.wrap_array(u_ode, semi)
    du = Trixi.wrap_array(du_ode, semi)

    drag_p = Trixi.analyze(drag_coefficient, du, u, tspan[2], mesh, equations, solver,
                           semi.cache, semi)
    lift_p = Trixi.analyze(lift_coefficient, du, u, tspan[2], mesh, equations, solver,
                           semi.cache, semi)

    drag_f = Trixi.analyze(drag_coefficient_shear_force, du, u, tspan[2], mesh,
                           equations, equations_parabolic, solver,
                           semi.cache, semi, semi.cache_parabolic)
    lift_f = Trixi.analyze(lift_coefficient_shear_force, du, u, tspan[2], mesh,
                           equations, equations_parabolic, solver,
                           semi.cache, semi, semi.cache_parabolic)

    @test isapprox(drag_p, 0.17963843913309516, atol = 1e-13)
    @test isapprox(lift_p, 0.26462588007949367, atol = 1e-13)

    @test isapprox(drag_f, 1.5427441885921553, atol = 1e-13)
    @test isapprox(lift_f, 0.005621910087395724, atol = 1e-13)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # We move these tests here to avoid modifying values used
    # to compute the drag/lift coefficients above.
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_NACA0012airfoil_mach085_restart.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_NACA0012airfoil_mach085_restart.jl"),
                        l2=[
                            6.18393805678897e-6,
                            0.00011581268033979716,
                            0.00011900457244798095,
                            0.006464813560434434
                        ],
                        linf=[
                            0.0017386607519724257,
                            0.07164832711649055,
                            0.03699801158450292,
                            1.4385095221954427
                        ], tspan=(0.0, 0.01))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_viscous_shock.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        l2=[
                            0.0002576236264053728,
                            0.00014336949098706463,
                            7.189100338239794e-17,
                            0.00017369905124642074
                        ],
                        linf=[
                            0.0016731940983241156,
                            0.0010638640749656147,
                            5.59044079947959e-16,
                            0.001149532023891009
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_viscous_shock.jl (boundary_condition_do_nothing)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        boundary_conditions_parabolic=(;
                                                       x_neg = boundary_condition_parabolic,
                                                       x_pos = boundary_condition_do_nothing),
                        l2=[
                            0.0002794565402113706,
                            0.00027552504911070345,
                            3.968443509704103e-16,
                            0.0005302753211545681
                        ],
                        linf=[
                            0.001673364743408845,
                            0.0010781000885169978,
                            3.639381241716835e-15,
                            0.002801091447666215
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_viscous_shock_newton_krylov.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock_newton_krylov.jl"),
                        tspan=(0.0, 0.1),
                        atol_lin_solve=1e-11,
                        rtol_lin_solve=1e-11,
                        atol_ode_solve=1e-10,
                        rtol_ode_solve=1e-10,
                        l2=[
                            3.428501006908931e-5,
                            2.5967418005884837e-5,
                            2.7084890458524478e-17,
                            2.855861765163304e-5
                        ],
                        linf=[
                            0.00018762342908784646,
                            0.0001405900207752664,
                            3.661971738081151e-16,
                            0.00014510700486747297
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_SD7003airfoil.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_SD7003airfoil.jl"),
                        l2=[
                            9.292899618740586e-5,
                            0.0001350510200255721,
                            7.964907891113045e-5,
                            0.0002336568736996096
                        ],
                        linf=[
                            0.2845637352223691,
                            0.295808392241858,
                            0.19309201225626166,
                            0.7188927326929244
                        ],
                        tspan=(0.0, 5e-3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_SD7003airfoil.jl (CFL-Interval)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_SD7003airfoil.jl"),
                        l2=[
                            9.292895651912815e-5,
                            0.0001350510066877861,
                            7.964905098170568e-5,
                            0.00023365678706785303
                        ],
                        linf=[
                            0.2845614660523972,
                            0.29577255454711177,
                            0.19307666048254143,
                            0.7188872358580256
                        ],
                        tspan=(0.0, 5e-3),
                        stepsize_callback=StepsizeCallback(cfl = 2.2, interval = 5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_RAE2822airfoil_separation.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_RAE2822airfoil_separation.jl"),
                        l2=[
                            4.825503683000652e-5,
                            5.9311607940124866e-5,
                            4.275960940632988e-5,
                            0.00013394505065590255
                        ],
                        linf=[
                            1.303552961776984,
                            1.1021933490327356,
                            0.8733558923919265,
                            3.6025810936812412
                        ],
                        tspan=(0.0, 5e-5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_vortex_street.jl (Re=20)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_vortex_street.jl"),
                        tspan=(0.0, 0.5),
                        Re=20, # Render flow diffusion-dominated
                        callbacks=CallbackSet(summary_callback, analysis_callback,
                                              alive_callback,
                                              StepsizeCallback(cfl = 2.3,
                                                               cfl_parabolic = 1.0)),
                        adaptive=false, # respect CFL
                        ode_alg=CKLLSRK95_4S(),
                        l2=[
                            0.011916725799140692,
                            0.027926098816747836,
                            0.01902700347912797,
                            0.11793406377747188
                        ],
                        linf=[
                            0.3546113252441576,
                            1.0152021857472098,
                            0.5811488174143082,
                            3.207373092525428
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_vortex_street.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_vortex_street.jl"),
                        l2=[
                            0.012420217727434794,
                            0.028935260981567217,
                            0.023078384429351353,
                            0.11317643179072025
                        ],
                        linf=[
                            0.4484833725983406,
                            1.268913882714608,
                            0.7071821629898418,
                            3.643975012834931
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_vortex_street.jl (GradientVariablesEntropy)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_vortex_street.jl"),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            0.01242797973116292,
                            0.02892502142448505,
                            0.0230829131666028,
                            0.11323126134096527
                        ],
                        linf=[
                            0.4544189333202735,
                            1.269315313304855,
                            0.7082067255956892,
                            3.6951068269010645
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_poiseuille_flow.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_poiseuille_flow.jl"),
                        l2=[
                            0.028671228188785286,
                            0.2136420195921885,
                            0.009953689550858224,
                            0.13216036594768157
                        ],
                        linf=[
                            0.30901218409540543,
                            1.3488655161645846,
                            0.1304661713119874,
                            1.2094591729756736],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_kelvin_helmholtz_instability_sc_subcell.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_kelvin_helmholtz_instability_sc_subcell.jl"),
                        l2=[
                            0.1987691550257618,
                            0.1003336666735962,
                            0.1599420846677608,
                            0.07314642823482713
                        ],
                        linf=[
                            0.8901520920065688,
                            0.47421178500575756,
                            0.38859478648621326,
                            0.3247497921546598
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs!, semi, sol, 15000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 20_500)
end

@testitem "Parabolic2D: elixir_navierstokes_freestream_symmetry.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_freestream_symmetry.jl"),
                        l2=[
                            4.37868326434923e-15,
                            7.002449644031901e-16,
                            1.0986677074164136e-14,
                            1.213800745067394e-14
                        ],
                        linf=[
                            2.531308496145357e-14,
                            3.8367543336926215e-15,
                            4.9960036108132044e-14,
                            6.705747068735946e-14
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_freestream_symmetry.jl (GradientVariablesEntropy)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_freestream_symmetry.jl"),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            4.37868326434923e-15,
                            7.002449644031901e-16,
                            1.0986677074164136e-14,
                            1.213800745067394e-14
                        ],
                        linf=[
                            2.531308496145357e-14,
                            3.8367543336926215e-15,
                            4.9960036108132044e-14,
                            6.705747068735946e-14
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_freestream_ldg.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_freestream_ldg.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            9.421780256792343e-17,
                            3.597722377997839e-15,
                            4.626519294383959e-15,
                            1.4160796757846855e-13
                        ],
                        linf=[
                            1.5543122344752192e-15,
                            1.2378986724570495e-13,
                            1.7111312367035225e-13,
                            1.5727863456049818e-11
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_couette_flow.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_couette_flow.jl"),
                        l2=[
                            0.009585252225488753,
                            0.007939233099864973,
                            0.0007617512688442657,
                            0.027229870237669436
                        ],
                        linf=[
                            0.027230029149270862,
                            0.027230451118692933,
                            0.0038642959675975713,
                            0.04738248734987671
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_blast_reflective.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_blast_reflective.jl"),
                        l2=[
                            0.013077652405653456,
                            0.03267271241679693,
                            0.03267271241679689,
                            0.19993587690609887
                        ],
                        linf=[
                            0.232863088636711,
                            0.5958991303183211,
                            0.5958991303183204,
                            3.0621202120365467
                        ],
                        tspan=(0.0, 0.01),
                        sol=solve(ode, ode_alg;
                                  adaptive = false, dt = 1e-4,
                                  ode_default_options()..., callback = callbacks))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_diffusion_2d.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_diffusion_2d.jl"),
                        initial_refinement_level=2,
                        l2=[0.0012719424527102708],
                        linf=[0.004796263597208306])
    @test Trixi.SciMLBase.successful_retcode(sol.retcode)
    coarse_l2_error, coarse_linf_error = analysis_callback(sol)

    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_diffusion_2d.jl"),
                        initial_refinement_level=3,
                        l2=[8.29287801975101e-5],
                        linf=[0.000657824071300106])
    @test Trixi.SciMLBase.successful_retcode(sol.retcode)
    fine_l2_error, fine_linf_error = analysis_callback(sol)
    @test all(fine_l2_error .< coarse_l2_error)
    @test all(fine_linf_error .< coarse_linf_error)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_diffusion_spatially_varying_diffusivity.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_diffusion_spatially_varying_diffusivity.jl"),
                        initial_refinement_level=2,
                        l2=[0.0019830194495154916],
                        linf=[0.007670700094495438])
    coarse_l2, coarse_linf = analysis_callback(sol)

    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_diffusion_spatially_varying_diffusivity.jl"),
                        initial_refinement_level=3,
                        l2=[0.0002087372225728981],
                        linf=[0.0009605379115259494])
    medium_l2, medium_linf = analysis_callback(sol)

    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_diffusion_spatially_varying_diffusivity.jl"),
                        initial_refinement_level=4,
                        l2=[2.6122940925001035e-5],
                        linf=[0.00011865187653337106])
    fine_l2, fine_linf = analysis_callback(sol)

    @test all(medium_l2 .< coarse_l2)
    @test all(fine_l2 .< medium_l2)
    @test all(medium_linf .< coarse_linf)
    @test all(fine_linf .< medium_linf)
    l2_orders = (log2.(coarse_l2 ./ medium_l2),
                 log2.(medium_l2 ./ fine_l2))
    linf_orders = (log2.(coarse_linf ./ medium_linf),
                   log2.(medium_linf ./ fine_linf))
    @test all(order -> all(order .>= polydeg - 0.1), l2_orders)
    @test all(order -> all(order .>= polydeg - 0.1), linf_orders)

    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_diffusion_steady_state_linear_map.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_diffusion_steady_state_linear_map.jl"),
                        tspan=(0.0, 1.0e-4),
                        analysis_callback=AnalysisCallback(semi,
                                                           interval = 1,
                                                           extra_analysis_errors = (:l2_error_primitive,
                                                                                    :linf_error_primitive),
                                                           extra_analysis_integrals = (entropy,)),
                        l2=[2.9029827892716424e-5], linf=[0.0003022506331279151],
                        # Relax error tols to avoid stochastic CI failures
                        atol=1e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end
