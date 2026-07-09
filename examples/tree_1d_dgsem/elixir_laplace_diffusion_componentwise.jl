using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of a component-wise diffusion equation

# Use a stationary passive-tracer Euler state as a multivariable carrier.
# Only the first tracer component has nonzero diffusivity.
flow_equations = CompressibleEulerEquations1D(1.4)
equations = PassiveTracerEquations(flow_equations, n_tracers = 2)
diffusivity() = SVector(0.0, 0.0, 0.0, 0.1, 0.0)
equations_parabolic = LaplaceDiffusionComponentwise1D(diffusivity(), equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_central)
solver_parabolic = ParabolicFormulationLocalDG()

mesh = TreeMesh(0.0, 2.0,
                initial_refinement_level = 4,
                n_cells_max = 30_000,
                periodicity = true)

function initial_condition_laplace_diffusion_componentwise(x, t, equations)
    rho = 1.0
    rho_v1 = 0.0
    rho_e_total = 1 / (equations.flow_equations.gamma - 1)
    kappa_tracer_1 = diffusivity()[4]
    rho_chi_1 = sinpi(x[1]) * exp(-kappa_tracer_1 * pi^2 * t)
    rho_chi_2 = 1 + 0.25 * sinpi(3 * x[1])
    return SVector(rho, rho_v1, rho_e_total, rho_chi_1, rho_chi_2)
end
initial_condition = initial_condition_laplace_diffusion_componentwise

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = solver_parabolic,
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 100)
alive_callback = AliveCallback(analysis_interval = 100)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL35();
            dt = 1.0e-4, adaptive = false,
            ode_default_options()..., callback = callbacks)
