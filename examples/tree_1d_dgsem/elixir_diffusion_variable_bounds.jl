using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the pure diffusion equation

diffusivity() = 0.5
equations = LinearDiffusionEquation1D(diffusivity())

# Create a DG solver with polynomial degree 3 and the local discontinuous Galerkin
# formulation for the parabolic operator.
solver = DGSEM(polydeg = 3)
solver_parabolic = ParabolicFormulationLocalDG()

# The periodic domain matches the wavelength of the manufactured sine profile.
mesh = TreeMesh(-Float64(pi), Float64(pi),
                initial_refinement_level = 4,
                n_cells_max = 30_000,
                periodicity = true)

# This exact solution of the heat equation remains in the interval [0, 1].
function initial_condition_positive_diffusion(x, t, equations)
    concentration = 0.5 +
                    0.5 * sin(x[1]) * exp(-diffusivity() * t)
    return SVector(concentration)
end
initial_condition = initial_condition_positive_diffusion

semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic = solver_parabolic,
                                   boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solver and callbacks

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 100)
alive_callback = AliveCallback(analysis_interval = 100)

# A monitored quantity can be any scalar function of the local state and equations.
concentration(u, equations) = u[1]
concentration_bound = VariableBound(:concentration, concentration;
                                    lower = 0.0,
                                    upper = 1.0,
                                    abstol = 1.0e-12,
                                    reltol = 1.0e-10)

# The callback records global nodal extrema without modifying the solution. It checks the
# initial state, every twentieth accepted step, and the final state.
variable_bounds_callback = VariableBoundsCallback(semi;
                                                  bounds = (concentration_bound,),
                                                  interval = 20,
                                                  check_initial = true,
                                                  check_final = true,
                                                  action = :warn,
                                                  save_analysis = false)

callbacks = CallbackSet(summary_callback, analysis_callback,
                        variable_bounds_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL35();
            dt = 1.0e-3, adaptive = false,
            ode_default_options()..., callback = callbacks)

# To terminate when a violation is detected, use:
#
# action = :terminate
