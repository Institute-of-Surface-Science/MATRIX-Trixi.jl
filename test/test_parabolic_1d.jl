@testsnippet Parabolic1D begin
    EXAMPLES_DIR = examples_dir()
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[8.40483031802723e-6],
                        linf=[2.8990878868540015e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_ldg.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_ldg.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[9.234438322146518e-6], linf=[5.425491770139068e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_ldg.jl (Gauss-Legendre)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_ldg.jl"),
                        solver=DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
                                     basis_type = GaussLegendreBasis),
                        tspan=(0.0, 0.4),
                        l2=[4.126471023759558e-6], linf=[1.4470099431229677e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_gradient_source_terms.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[1.0990454698899562e-5], linf=[6.469747978055107e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_restart.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_restart.jl"),
                        l2=[1.0679933947301556e-5],
                        linf=[3.910500545667439e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_cfl.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_cfl.jl"),
                        l2=[6.763177530985864e-5], linf=[0.0002344578097126515])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_dirichlet_amr.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_dirichlet_amr.jl"),
                        l2=[3.668679081538521e-6], linf=[0.0001053981743872842])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_neumann_amr.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_neumann_amr.jl"),
                        l2=[0.9974473329813947], linf=[1.0000064761980827])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion.jl (AMR)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        tspan=(0.0, 0.0), initial_refinement_level=5)
    tspan = (0.0, 1.0)
    ode = semidiscretize(semi, tspan)
    amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                          base_level = 4,
                                          med_level = 5, med_threshold = 0.1,
                                          max_level = 6, max_threshold = 0.6)
    amr_callback = AMRCallback(semi, amr_controller,
                               interval = 5,
                               adapt_initial_condition = true)

    # Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
    callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                            amr_callback)
    sol = solve(ode, ode_alg;
                abstol = time_abs_tol, reltol = time_int_tol,
                ode_default_options()..., callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈ [6.487940740394583e-6]
    @test linf_error ≈ [3.262867898701227e-5]
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_implicit_sparse_jacobian.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_implicit_sparse_jacobian.jl"),
                        tspan=(0.0, 0.4),
                        l2=[0.05240130204342638], linf=[0.07407444680136666])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_implicit_sparse_jacobian_restart.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_implicit_sparse_jacobian_restart.jl"),
                        l2=[0.08292233849124372], linf=[0.11726345328639576])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: elixir_advection_implicit_sparse_jacobian_restart.jl (no colorvec)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_implicit_sparse_jacobian_restart.jl"),
                        colorvec_parabolic=nothing,
                        l2=[0.08292233849124372], linf=[0.11726345328639576])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_periodic.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        l2=[
                            0.0001133835907077494,
                            6.226282245610444e-5,
                            0.0002820171699999139
                        ],
                        linf=[
                            0.0006255102377159538,
                            0.00036195501456059986,
                            0.0016147729485886941
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_periodic_cfl.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_periodic_cfl.jl"),
                        l2=[
                            0.00011582226718630047,
                            6.277345250542003e-5,
                            0.0002822257163816253
                        ],
                        linf=[
                            0.0006389893469918029,
                            0.0003608325914101762,
                            0.0016369657641206459
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_periodic.jl: GradientVariablesEntropy" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            0.00011310615871043463,
                            6.216495207074201e-5,
                            0.00028195843110817814
                        ],
                        linf=[
                            0.0006240837363233886,
                            0.0003616694320713876,
                            0.0016147339542413874
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_walls.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls.jl"),
                        l2=[
                            0.0004702331100298379,
                            0.0003218173539588441,
                            0.001496626616191212
                        ],
                        linf=[
                            0.0029963751636357117,
                            0.0028639041695096433,
                            0.012691132694550689
                        ],
                        atol=1e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_walls.jl: GradientVariablesEntropy" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            0.00046085004909354776,
                            0.0003243109084492897,
                            0.0015159733164383632
                        ],
                        linf=[
                            0.0027548031865172184,
                            0.0028567713569609024,
                            0.012941793735691931
                        ],
                        atol=1e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_walls.jl (Gauss-Legendre)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls.jl"),
                        solver=DGSEM(polydeg = 3, surface_flux = flux_hll,
                                     basis_type = GaussLegendreBasis),
                        time_int_tol=1e-10,
                        l2=[
                            4.201445769104007e-5,
                            9.758279535510314e-5,
                            0.0004199990641561288
                        ],
                        linf=[
                            0.00015356293659607445,
                            0.0004198436005902785,
                            0.0016946745322332646
                        ],
                        atol=1e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_walls_amr.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls_amr.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number()),
                        l2=[
                            2.5278845598681636e-5,
                            2.5540145802666872e-5,
                            0.0001211867535580826
                        ],
                        linf=[
                            0.0001466387202588848,
                            0.00019422419092429135,
                            0.0009556449835592673
                        ],
                        atol=1e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_walls_amr.jl: GradientVariablesEntropy" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls_amr.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            2.4593521887223632e-5,
                            2.3928212900127102e-5,
                            0.00011252332663824173
                        ],
                        linf=[
                            0.00011850494672183132,
                            0.00018987676556476442,
                            0.0009597423024825247
                        ],
                        atol=1e-8)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_viscous_shock.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        l2=[
                            0.00025762354103445303,
                            0.0001433692781569829,
                            0.00017369861968287976
                        ],
                        linf=[
                            0.0016731940030498826,
                            0.0010638575921477766,
                            0.0011495207677434394
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_viscous_shock.jl (Gauss-Legendre)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        solver=DGSEM(polydeg = 3, surface_flux = flux_hlle,
                                     basis_type = GaussLegendreBasis),
                        l2=[
                            0.00010415910094963455,
                            7.569570282227496e-5,
                            8.643799824895884e-5
                        ],
                        linf=[
                            0.0004795456761867989,
                            0.0003525509032139551,
                            0.0004044657250887873
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_viscous_shock.jl (boundary_condition_do_nothing)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        boundary_conditions_parabolic=(;
                                                       x_neg = boundary_condition_parabolic,
                                                       x_pos = boundary_condition_do_nothing),
                        l2=[
                            0.00027945595319833104,
                            0.00027552386931121406,
                            0.0005302742561529139
                        ],
                        linf=[
                            0.0016733632873879856,
                            0.001078100012167113,
                            0.0028010908633919196
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_viscous_shock_imex.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_viscous_shock_imex.jl"),
                        atol_lin_solve=1e-11, rtol_lin_solve=1e-10,
                        l2=[
                            0.0016637028933384878,
                            0.0014571255711373966,
                            0.0014843783212282159
                        ],
                        linf=[
                            0.00545660697650141,
                            0.003950431201790283,
                            0.004092051414554598
                        ],
                        # Relax error tols to avoid stochastic CI failures
                        atol=1e-7)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_viscous_burgers_n_wave.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_viscous_burgers_n_wave.jl"),
                        l2=[0.03005971517609335], linf=[0.08174614630359545])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_viscous_burgers_shock.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_viscous_burgers_shock.jl"),
                        l2=[0.0025484696686361645], linf=[0.028069313915933147])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: DGMulti: elixir_advection_diffusion_gradient_source_terms.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        l2=[0.01889578192611483],
                        linf=[0.03572728414418691])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: DGMulti: elixir_advection_diffusion_sbp.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_advection_diffusion_sbp.jl"),
                        l2=[2.027026825559297e-5],
                        linf=[3.1997648799242384e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: DGMulti: elixir_navierstokes_convergence_periodic.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        l2=[
                            3.7943372542675425e-5,
                            4.078766566292102e-5,
                            0.00024524952267207235
                        ],
                        linf=[
                            0.00010969455084941515,
                            9.183113730193426e-5,
                            0.0005450421812014383
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "Parabolic1D: DGMulti: elixir_navierstokes_convergence_periodic.jl (Diff. CFL)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        callbacks=CallbackSet(summary_callback, alive_callback,
                                              analysis_callback,
                                              StepsizeCallback(cfl = 0.5,
                                                               cfl_parabolic = 0.1)),
                        adaptive=false,
                        l2=[
                            3.804624387087144e-5,
                            4.0776239664045585e-5,
                            0.0002452796554181002
                        ],
                        linf=[
                            0.00010899905841177393,
                            9.108558032178138e-5,
                            0.0005277952647766426
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: DGMulti: elixir_navierstokes_convergence_periodic.jl (GradientVariablesEntropy)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            3.855011159752911e-5,
                            4.077230736638483e-5,
                            0.0002457818746735199
                        ],
                        linf=[
                            0.00011052974882530542,
                            9.179337892284423e-5,
                            0.00054534178933352
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_reaction_diffusion_immobile_species_imex.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_reaction_diffusion_immobile_species_imex.jl"),
                        l2=[2.1610180405591028e-7, 4.193868403256278e-7],
                        linf=[5.043982367336497e-7, 9.672281314765385e-7])
    @test Trixi.SciMLBase.successful_retcode(sol.retcode)
    l2_error, linf_error = analysis_callback(sol)
    @test maximum(l2_error) < 1.0e-6
    @test maximum(linf_error) < 1.0e-6
    @test_nowarn linear_structure(semi)

    reaction_disabled_mode = evolve_reaction_diffusion_mode(1.0, 0.25, 0, 0.1;
                                                            forward_rate = 0.0,
                                                            backward_rate = 0.0)
    @test reaction_disabled_mode == SVector(1.0, 0.25)

    mass_initial = Trixi.integrate(sol.u[1], semi; normalize = false)
    mass_final = Trixi.integrate(sol.u[end], semi; normalize = false)
    @test isapprox(sum(mass_final), sum(mass_initial); atol = 1.0e-10, rtol = 1.0e-10)

    du_explicit = similar(sol.u[end])
    Trixi.rhs!(du_explicit, sol.u[end], semi, sol.t[end])
    @test iszero(maximum(abs, du_explicit))

    semi_no_reaction = remake(semi; source_terms_parabolic = nothing)
    ode_no_reaction = semidiscretize(semi_no_reaction, tspan)
    mobile_bound = VariableBound(:mobile, (u, equations) -> u[1];
                                 lower = 0.0, upper = 2.0)
    immobile_bound = VariableBound(:immobile, (u, equations) -> u[2];
                                   lower = 0.0, upper = 1.0)
    imex_bounds_callback = VariableBoundsCallback(semi_no_reaction;
                                                  bounds = (mobile_bound,
                                                            immobile_bound),
                                                  interval = 1,
                                                  action = :record)
    sol_no_reaction = solve(ode_no_reaction, ode_alg;
                            abstol = 1.0e-8, reltol = 1.0e-8,
                            save_everystep = false,
                            callback = imex_bounds_callback)
    @test Trixi.SciMLBase.successful_retcode(sol_no_reaction.retcode)
    @test imex_bounds_callback.affect!.checks_performed ==
          sol_no_reaction.stats.naccept + 1
    @test !isviolated(imex_bounds_callback.affect!.last_results.mobile)
    @test !isviolated(imex_bounds_callback.affect!.last_results.immobile)

    u_initial = Trixi.wrap_array(sol_no_reaction.u[1], semi_no_reaction)
    u_final = Trixi.wrap_array(sol_no_reaction.u[end], semi_no_reaction)
    mobile_change = maximum(abs, @view(u_final[1, :, :]) .-
                                 @view(u_initial[1, :, :]))
    immobile_change = maximum(abs, @view(u_final[2, :, :]) .-
                                   @view(u_initial[2, :, :]))
    @test mobile_change > 1.0e-5
    @test immobile_change < 1.0e-13

    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_laplace_diffusion_componentwise.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_laplace_diffusion_componentwise.jl"),
                        l2=[3.15152149243869e-15, 3.4751577256217856e-16,
                            4.387506658169924e-15, 1.0219956109529742e-5,
                            0.00011135716421812636],
                        linf=[3.219646771412954e-15, 5.003097405398211e-16,
                            4.884981308350689e-15, 5.97916749279781e-5,
                            0.0002426087236215846])
    @test Trixi.SciMLBase.successful_retcode(sol.retcode)

    u_initial = Trixi.wrap_array(sol.u[1], semi)
    u_final = Trixi.wrap_array(sol.u[end], semi)
    for variable in (1, 2, 3, 5)
        @test maximum(abs,
                      @view(u_final[variable, :, :]) .-
                      @view(u_initial[variable, :, :])) < 1.0e-13
    end

    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: DGMulti componentwise diffusion RHS" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    dg = DGMulti(polydeg = 2, element_type = Line(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central))
    mesh = DGMultiMesh(dg, (2,), coordinates_min = (0.0,), coordinates_max = (2.0,),
                       periodicity = true)
    equations = CompressibleEulerEquations1D(1.4)
    equations_parabolic = LaplaceDiffusionComponentwise1D((0.1, 0.0, 0.0), equations)
    initial_condition = function (x, t, equations)
        return SVector(1 + 0.1 * sinpi(x[1]), zero(x[1]),
                       one(x[1]) / (equations.gamma - 1))
    end

    semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition, dg;
                                                 boundary_conditions = (boundary_condition_periodic,
                                                                        boundary_condition_periodic))
    ode = semidiscretize(semi, (0.0, 0.01))
    du = similar(ode.u0)
    @test_nowarn Trixi.rhs_parabolic!(du, ode.u0, semi, 0.0)

    du_fields = Base.parent(du)
    @test maximum(abs, getindex.(du_fields, 1)) > 1.0e-8
    for variable in 2:nvariables(equations)
        @test iszero(maximum(abs, getindex.(du_fields, variable)))
    end
end

@testitem "Parabolic1D: TreeMesh1D: elixir_diffusion_ldg.jl" setup=[Setup, Parabolic1D] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[9.235894939144276e-6], linf=[5.402550135213957e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_diffusion_boundary_flux.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_boundary_flux.jl"),
                        tspan=(0.0, 0.001))

    t = sol.t[end]
    u_ode = sol.u[end]
    du_ode = similar(u_ode)
    Trixi.rhs_parabolic!(du_ode, u_ode, semi, t)
    u = Trixi.wrap_array(u_ode, semi)
    du = Trixi.wrap_array(du_ode, semi)

    flux_left = Trixi.analyze(normal_flux_left, du, u, t, semi)
    flux_right = Trixi.analyze(normal_flux_right, du, u, t, semi)
    flux_total = Trixi.analyze(normal_flux_total, du, u, t, semi)
    physical_right = Trixi.analyze(fick_outflow_right, du, u, t, semi)

    @test flux_left≈0.5 atol=1.0e-11 rtol=1.0e-11
    @test flux_right≈-0.5 atol=1.0e-11 rtol=1.0e-11
    @test flux_total≈0.0 atol=1.0e-11
    @test physical_right≈0.5 atol=1.0e-11 rtol=1.0e-11

    boundary_conditions_neumann = (;
                                   x_neg = BoundaryConditionNeumann((x, t, equations) -> SVector(0.25)),
                                   x_pos = BoundaryConditionNeumann((x, t, equations) -> SVector(-0.75)))
    semi_neumann = remake(semi; boundary_conditions = boundary_conditions_neumann)
    du_neumann_ode = similar(u_ode)
    Trixi.rhs_parabolic!(du_neumann_ode, u_ode, semi_neumann, t)
    u_neumann = Trixi.wrap_array(u_ode, semi_neumann)
    du_neumann = Trixi.wrap_array(du_neumann_ode, semi_neumann)

    neumann_flux_left = Trixi.analyze(normal_flux_left, du_neumann, u_neumann, t,
                                      semi_neumann)
    neumann_flux_right = Trixi.analyze(normal_flux_right, du_neumann, u_neumann, t,
                                       semi_neumann)
    neumann_flux_total = neumann_flux_left + neumann_flux_right
    integrated_rhs = Trixi.integrate(du_neumann_ode, semi_neumann; normalize = false)

    @test neumann_flux_left≈0.25 atol=1.0e-11 rtol=1.0e-11
    @test neumann_flux_right≈-0.75 atol=1.0e-11 rtol=1.0e-11
    @test neumann_flux_total≈-0.5 atol=1.0e-11 rtol=1.0e-11
    @test integrated_rhs[1]≈neumann_flux_total atol=1.0e-11 rtol=1.0e-11

    invalid_component = AnalysisSurfaceIntegral((:x_pos,), NormalParabolicFlux(2))
    @test_throws ArgumentError Trixi.analyze(invalid_component, du, u, t, semi)
    invalid_boundary = AnalysisSurfaceIntegral((:not_a_boundary,), NormalParabolicFlux())
    @test_throws ArgumentError Trixi.analyze(invalid_boundary, du, u, t, semi)

    @test Trixi.pretty_form_ascii(normal_flux_left) == "normal_flux_left"
    @test Trixi.pretty_form_utf(fick_outflow_right) == "fick_outflow_right"
    @test NormalParabolicFlux(Int32(1)) isa NormalParabolicFlux{1}
    @test NormalParabolicFlux(big(1)) isa NormalParabolicFlux{1}
    @test_throws ArgumentError NormalParabolicFlux(0)
end

@testitem "Parabolic1D: TreeMesh diffusion boundary flux with Gauss nodes" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_boundary_flux.jl"),
                        solver=DGSEM(polydeg = 3, basis_type = GaussLegendreBasis),
                        tspan=(0.0, 0.001))

    t = sol.t[end]
    du_ode = similar(sol.u[end])
    Trixi.rhs_parabolic!(du_ode, sol.u[end], semi, t)
    u = Trixi.wrap_array(sol.u[end], semi)
    du = Trixi.wrap_array(du_ode, semi)
    @test Trixi.analyze(normal_flux_left, du, u, t, semi)≈0.5 atol=1.0e-10 rtol=1.0e-10
    @test Trixi.analyze(normal_flux_right, du, u, t, semi)≈-0.5 atol=1.0e-10 rtol=1.0e-10
end

@testitem "Parabolic1D: TreeMesh diffusion boundary flux with BR1" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_boundary_flux.jl"),
                        solver_parabolic=ParabolicFormulationBassiRebay1(),
                        tspan=(0.0, 0.001))

    t = sol.t[end]
    du_ode = similar(sol.u[end])
    Trixi.rhs_parabolic!(du_ode, sol.u[end], semi, t)
    u = Trixi.wrap_array(sol.u[end], semi)
    du = Trixi.wrap_array(du_ode, semi)
    @test Trixi.analyze(normal_flux_left, du, u, t, semi)≈0.5 atol=1.0e-11 rtol=1.0e-11
    @test Trixi.analyze(normal_flux_right, du, u, t, semi)≈-0.5 atol=1.0e-11 rtol=1.0e-11
end

@testitem "Parabolic1D: TreeMesh1D: elixir_diffusion_time_dependent_coefficients.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_time_dependent_coefficients.jl"),
                        l2=[6.984694319086482e-6],
                        linf=[4.066890902476583e-5])
    @test Trixi.SciMLBase.successful_retcode(sol.retcode)
    fine_l2_error, fine_linf_error = analysis_callback(sol)
    @test all(fine_l2_error .< [0.00011127464521867602])
    @test all(fine_linf_error .< [0.0006347956752195128])
    @test have_space_time_dependent_flux(equations) == Trixi.True()

    semi_flux_only = remake(semi; source_terms = nothing)
    u0 = copy(ode.u0)
    du_t0 = similar(u0)
    du_t1 = similar(u0)
    Trixi.rhs_parabolic!(du_t0, u0, semi_flux_only, 0.0)
    Trixi.rhs_parabolic!(du_t1, u0, semi_flux_only, 0.4)
    @test du_t0 != du_t1
    @test sqrt(sum(abs2, du_t0 .- du_t1)) > 100 * eps(eltype(u0))

    gradients = (SVector(1.0),)
    x = SVector(0.0)
    @test Trixi.flux(SVector(1.0), gradients, 1, x, 0.0, equations) isa SVector{1}
    @test Trixi.flux(SVector(1.0), gradients, 1, x, 0.0, equations) !=
          Trixi.flux(SVector(1.0), gradients, 1, x, 0.5, equations)

    monotonic_diffusivity_value = (x, t, equations) -> 0.1 * (1 + t)
    monotonic_diffusivity = SpatiallyVaryingDiffusivity(monotonic_diffusivity_value, 0.15)
    equations_monotonic = LinearDiffusionEquation1D(monotonic_diffusivity)
    semi_monotonic = remake(semi; equations = equations_monotonic)
    ode_monotonic = semidiscretize(semi_monotonic, tspan)
    u_monotonic = Trixi.wrap_array(ode_monotonic.u0, semi_monotonic)
    t_initial, t_final = tspan
    dt_initial = Trixi.max_dt(u_monotonic, t_initial, mesh,
                              have_constant_diffusivity(equations_monotonic),
                              equations_monotonic, equations_monotonic, solver,
                              semi_monotonic.cache)
    dt_final = Trixi.max_dt(u_monotonic, t_final, mesh,
                            have_constant_diffusivity(equations_monotonic),
                            equations_monotonic, equations_monotonic, solver,
                            semi_monotonic.cache)
    @test dt_final < dt_initial
    @test dt_final ≈ dt_initial / (1 + t_final)

    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: Time-dependent coefficients refinement" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_time_dependent_coefficients.jl"),
                        initial_refinement_level=3,
                        l2=[0.00011127464521867602],
                        linf=[0.0006347956752195128])
    @test Trixi.SciMLBase.successful_retcode(sol.retcode)
    coarse_l2_error, coarse_linf_error = analysis_callback(sol)
    @test all([6.984694319086482e-6] .< coarse_l2_error)
    @test all([4.066890902476583e-5] .< coarse_linf_error)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_diffusion_ldg_newton_krylov.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg_newton_krylov.jl"),
                        atol_lin_solve=1e-11, rtol_lin_solve=1e-10,
                        atol_ode_solve=1e-10, rtol_ode_solve=1e-9,
                        l2=[4.14999791227157e-6], linf=[2.424658410971059e-5],
                        # Relax error tols to avoid stochastic CI failures
                        atol=1e-12)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_diffusion_ldg_implicit.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg_implicit.jl"),
                        initial_refinement_level=3,
                        l2=[6.551916685521166e-5],
                        linf=[0.0004907349680308348])
    @test Trixi.SciMLBase.successful_retcode(sol.retcode)
    coarse_l2_error, coarse_linf_error = analysis_callback(sol)

    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg_implicit.jl"),
                        initial_refinement_level=4,
                        l2=[5.938549329272694e-6],
                        linf=[6.25104098415639e-5])
    @test Trixi.SciMLBase.successful_retcode(sol.retcode)
    fine_l2_error, fine_linf_error = analysis_callback(sol)
    @test all(fine_l2_error .< coarse_l2_error)
    @test all(fine_linf_error .< coarse_linf_error)

    scalar_bound = VariableBound(:scalar, (u, equations) -> u[1];
                                 lower = -0.1, upper = 1.1)
    implicit_bounds_callback = VariableBoundsCallback(semi;
                                                      bounds = (scalar_bound,),
                                                      interval = 1,
                                                      action = :record)
    implicit_ode = semidiscretize(semi, (0.0, 0.01))
    implicit_solution = solve(implicit_ode,
                              TRBDF2(; autodiff = AutoFiniteDiff());
                              abstol = time_int_tol, reltol = time_int_tol,
                              dt = 1.0e-2, save_everystep = false,
                              callback = implicit_bounds_callback)
    @test Trixi.SciMLBase.successful_retcode(implicit_solution.retcode)
    @test implicit_bounds_callback.affect!.checks_performed ==
          implicit_solution.stats.naccept + 1
    @test !isviolated(implicit_bounds_callback.affect!.last_results.scalar)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: VariableBoundsCallback" setup=[Setup, Parabolic1D] tags=[
    :parabolic_part1
] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_variable_bounds.jl"),
                        l2=[5.326102683288938e-6],
                        linf=[3.122527489207716e-5])

    results = variable_bounds_callback(sol)
    @test !isviolated(results.concentration)
    @test results.concentration.minimum >= -1.0e-12
    @test results.concentration.maximum <= 1.0 + 1.0e-12
    @test variable_bounds_callback.affect!.checks_performed == 6
    @test variable_bounds_callback.affect!.violations_detected == 0

    @test_throws ArgumentError VariableBound(:invalid, concentration)
    @test_throws ArgumentError VariableBound(:invalid, concentration;
                                             lower = 1.0, upper = 0.0)
    @test_throws ArgumentError VariableBound(:invalid, concentration;
                                             lower = 0.0, abstol = -1.0)
    @test_throws ArgumentError VariableBound(:invalid, concentration;
                                             upper = 1.0, reltol = -1.0)
    @test_throws ArgumentError VariableBound(:invalid, concentration; lower = NaN)
    @test_throws ArgumentError VariableBound(:invalid, concentration; upper = Inf)
    @test_throws ArgumentError VariableBound(:invalid, concentration;
                                             lower = 0.0, abstol = Inf)
    @test_throws ArgumentError VariableBound(:invalid, concentration;
                                             upper = 1.0, reltol = Inf)
    @test_throws ArgumentError VariableBoundsCallback(semi;
                                                      bounds = (concentration_bound,),
                                                      interval = -1)
    @test_throws ArgumentError VariableBoundsCallback(semi;
                                                      bounds = (concentration_bound,),
                                                      action = :invalid)
    @test_throws ArgumentError VariableBoundsCallback(semi;
                                                      bounds = (concentration_bound,
                                                                concentration_bound))
    noncallable_bound = VariableBound(:noncallable, 42; lower = 0.0)
    @test_throws ArgumentError VariableBoundsCallback(semi;
                                                      bounds = (noncallable_bound,))
    complex_bound = VariableBound(:complex, (u, equations) -> complex(u[1]);
                                  lower = 0.0)
    @test_throws ArgumentError VariableBoundsCallback(semi; bounds = (complex_bound,))

    lower_bound = VariableBound(:lower, concentration; lower = 0.0)
    lower_result = Trixi.make_bounds_result(lower_bound, -0.1, 0.8, 2, 0)
    @test lower_result.minimum == -0.1
    @test lower_result.maximum == 0.8
    @test lower_result.lower_violation == 0.1
    @test lower_result.lower_violated
    @test !lower_result.upper_violated

    upper_bound = VariableBound(:upper, concentration; upper = 1.0)
    upper_result = Trixi.make_bounds_result(upper_bound, 0.1, 1.2, 2, 0)
    @test upper_result.upper_violation ≈ 0.2
    @test upper_result.upper_violated

    tolerated_bound = VariableBound(:tolerated, concentration;
                                    lower = 0.0, abstol = 1.0e-12)
    tolerated_result = Trixi.make_bounds_result(tolerated_bound, -1.0e-13,
                                                0.8, 2, 0)
    @test tolerated_result.lower_violation == 1.0e-13
    @test !tolerated_result.lower_violated
    violated_result = Trixi.make_bounds_result(tolerated_bound, -1.0e-8,
                                               0.8, 2, 0)
    @test violated_result.lower_violated

    relative_bound = VariableBound(:relative, concentration;
                                   lower = 100.0, reltol = 1.0e-2)
    relative_tolerated = Trixi.make_bounds_result(relative_bound, 99.5, 101.0,
                                                  2, 0)
    relative_violated = Trixi.make_bounds_result(relative_bound, 98.0, 101.0,
                                                 2, 0)
    @test relative_tolerated.lower_violation == 0.5
    @test !relative_tolerated.lower_violated
    @test relative_violated.lower_violated

    precise_lower = Float64(1) + 1.0e-8
    precise_bound = VariableBound(:precise, concentration; lower = precise_lower)
    precise_result = Trixi.make_bounds_result(precise_bound, Float32(1), Float32(1),
                                              1, 0)
    @test precise_result.minimum isa Float64
    @test precise_result.lower_violation ≈ 1.0e-8
    @test precise_result.lower_violated

    nonlinear_bound = VariableBound(:concentration_squared,
                                    (u, equations) -> u[1]^2;
                                    lower = 0.0, upper = 1.0)
    initial_state = copy(ode.u0)
    nonlinear_results = Trixi.evaluate_variable_bounds(ode.u0, semi,
                                                       (nonlinear_bound,))
    @test nonlinear_results.concentration_squared.minimum >= 0.0
    @test nonlinear_results.concentration_squared.maximum <= 1.0
    @test ode.u0 == initial_state

    nonfinite_state = copy(ode.u0)
    nonfinite_state[1] = NaN
    nonfinite_state[2] = Inf
    nonfinite_results = Trixi.evaluate_variable_bounds(nonfinite_state, semi,
                                                       (concentration_bound,))
    @test nonfinite_results.concentration.nonfinite_count == 2
    @test isviolated(nonfinite_results.concentration)

    fill!(nonfinite_state, NaN)
    all_nonfinite_results = Trixi.evaluate_variable_bounds(nonfinite_state, semi,
                                                           (concentration_bound,))
    @test all_nonfinite_results.concentration.finite_count == 0
    @test isnan(all_nonfinite_results.concentration.minimum)
    @test isnan(all_nonfinite_results.concentration.maximum)

    fsal_ode = semidiscretize(semi, (0.0, 0.01))
    solution_without_bounds = solve(fsal_ode, RDPK3SpFSAL35();
                                    dt = 1.0e-3, adaptive = false,
                                    ode_default_options()...)
    fsal_callback = VariableBoundsCallback(semi;
                                           bounds = (concentration_bound,),
                                           interval = 1,
                                           check_initial = false,
                                           check_final = false,
                                           action = :record)
    solution_with_bounds = solve(fsal_ode, RDPK3SpFSAL35();
                                 dt = 1.0e-3, adaptive = false,
                                 ode_default_options()...,
                                 callback = fsal_callback)
    @test solution_with_bounds.stats.nf == solution_without_bounds.stats.nf

    adaptive_callback = VariableBoundsCallback(semi;
                                               bounds = (concentration_bound,),
                                               interval = 2,
                                               check_initial = true,
                                               check_final = true,
                                               action = :record)
    adaptive_solution = solve(ode, RDPK3SpFSAL35();
                              dt = 0.1, adaptive = true,
                              abstol = 1.0e-12, reltol = 1.0e-12,
                              ode_default_options()...,
                              callback = adaptive_callback)
    expected_checks = 1 + adaptive_solution.stats.naccept ÷ 2 +
                      !iszero(adaptive_solution.stats.naccept % 2)
    @test adaptive_solution.stats.nreject > 0
    @test adaptive_callback.affect!.checks_performed == expected_checks

    termination_bound = VariableBound(:impossible, concentration; lower = 2.0)
    termination_callback = VariableBoundsCallback(semi;
                                                  bounds = (termination_bound,),
                                                  interval = 1,
                                                  check_initial = false,
                                                  action = :terminate)
    termination_solution = solve(ode, RDPK3SpFSAL35();
                                 dt = 1.0e-3, adaptive = false,
                                 ode_default_options()...,
                                 callback = termination_callback)
    @test termination_solution.t[end] < last(tspan)
    @test termination_callback.affect!.violations_detected == 1
    @test isviolated(termination_callback.affect!.last_results.impossible)

    initial_termination_callback = VariableBoundsCallback(semi;
                                                          bounds = (termination_bound,),
                                                          interval = 0,
                                                          check_initial = true,
                                                          action = :terminate)
    initial_termination_solution = solve(ode, RDPK3SpFSAL35();
                                         dt = 1.0e-3, adaptive = false,
                                         ode_default_options()...,
                                         callback = initial_termination_callback)
    @test initial_termination_solution.t[end] == first(tspan)
    @test initial_termination_callback.affect!.checks_performed == 1

    mktempdir() do output_directory
        file_callback = VariableBoundsCallback(semi;
                                               bounds = (concentration_bound,),
                                               interval = 0,
                                               check_initial = true,
                                               check_final = true,
                                               action = :record,
                                               save_analysis = true,
                                               output_directory = output_directory)
        solve(ode, RDPK3SpFSAL35();
              dt = 1.0e-3, adaptive = false,
              ode_default_options()..., callback = file_callback)
        filename = joinpath(output_directory, "variable_bounds.dat")
        lines_first_run = readlines(filename)
        @test isfile(filename)
        @test count(line -> startswith(line, "#"), lines_first_run) == 1
        @test count(line -> occursin("concentration", line), lines_first_run) == 2
        @test startswith(lines_first_run[2], "0 ")

        solve(ode, RDPK3SpFSAL35();
              dt = 1.0e-3, adaptive = false,
              ode_default_options()..., callback = file_callback)
        @test length(readlines(filename)) == length(lines_first_run)
    end

    @test_nowarn show(devnull, variable_bounds_callback)
    @test_nowarn show(devnull, MIME"text/plain"(), variable_bounds_callback)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_diffusion_ldg_amr_boundary_layer.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg_amr_boundary_layer.jl"),
                        l2=[0.5881457102264551], linf=[0.9302621795999283])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)

    # Test `show` method not exercised in elixirs
    @trixi_test_nowarn show(IOContext(stdout, :compact => true), MIME"text/plain"(),
                            semi)

    # Test basic semidiscretization dispatches
    @test ndims(semi) == ndims(semi.mesh)
    @test nvariables(semi) == nvariables(semi.equations)
    @test real(semi) == real(semi.solver)

    # Test that `remake` works for `SemidiscretizationParabolic`
    semi_remade = remake(semi)
    @test semi_remade isa SemidiscretizationParabolic
    @test semi_remade !== semi
    @test semi_remade.mesh === semi.mesh
    @test Trixi.ndofsglobal(semi_remade) == Trixi.ndofsglobal(semi)
end

@testitem "Parabolic1D: TreeMesh1D consistency check: elixir_diffusion_ldg_dirichlet.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    # Run the Dirichlet-Dirichlet elixir (uses `SemidiscretizationParabolic`)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg_dirichlet.jl"),
                        tspan=(0.0, 0.1),
                        analysis_callback=AnalysisCallback(semi,
                                                           interval = 100,
                                                           extra_analysis_errors = (:l2_error_primitive,
                                                                                    :linf_error_primitive),
                                                           extra_analysis_integrals = (entropy,)),
                        l2=[2.3481439150004898e-6], linf=[2.4576876189230656e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)

    # Store reference solution for comparison
    reference_solution = copy(sol.u[end])

    # Run again using an advection-diffusion equation with advection velocity zero
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg_dirichlet.jl"),
                        tspan=(0.0, 0.1),
                        equations=LinearScalarAdvectionEquation1D(0.0),
                        semi=SemidiscretizationHyperbolicParabolic(mesh,
                                                                   (equations,
                                                                    LaplaceDiffusion1D(diffusivity(),
                                                                                       equations)),
                                                                   initial_condition,
                                                                   solver;
                                                                   solver_parabolic = solver_parabolic,
                                                                   boundary_conditions = (boundary_conditions,
                                                                                          boundary_conditions)))
    # Check if the solutions for `SemidiscretizationParabolic` match those from 
    # `SemidiscretizationHyperbolicParabolic` using the same Float64 tolerance defaults as
    # `@test_trixi_include` in TrixiTest.jl.
    @test sol.u[end]≈reference_solution atol=500 * eps(Float64) rtol=sqrt(eps(Float64))
end
