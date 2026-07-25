using ADTypes
using OrdinaryDiffEqSDIRK
using Trixi

###############################################################################
# semidiscretization of the pure diffusion equation

diffusivity() = 0.5
equations = LinearDiffusionEquation1D(diffusivity())

polydeg = 3
solver = DGSEM(polydeg = polydeg)
solver_parabolic = ParabolicFormulationLocalDG()

initial_refinement_level = 4
mesh = TreeMesh(0.0, 1.0,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 30_000,
                periodicity = false)

# The initial state is incompatible with the instantaneous perfect-sink boundary data.
initial_condition_uniform(x, t, equations) = SVector(1.0)
perfect_sink(x, t, equations) = SVector(0.0)
initial_condition = initial_condition_uniform
boundary_conditions = (; x_neg = BoundaryConditionDirichlet(perfect_sink),
                       x_pos = BoundaryConditionDirichlet(perfect_sink))

semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic = solver_parabolic,
                                   boundary_conditions = boundary_conditions)

###############################################################################
# ODE solver and callbacks

tspan = (0.0, 0.01)
ode = semidiscretize(semi, tspan)

concentration(u, equations) = u[1]
lower = (0.0,)
upper = (1.0,)
variables = (concentration,)
limiter! = BoundsPreservingLimiterZhangShu(; lower, upper, variables)

concentration_bound = VariableBound(:concentration, concentration;
                                    lower = first(lower), upper = first(upper),
                                    abstol = 1.0e-12)
variable_bounds_callback = VariableBoundsCallback(semi;
                                                  bounds = (concentration_bound,),
                                                  interval = 1,
                                                  action = :record)

summary_callback = SummaryCallback()
callbacks = CallbackSet(summary_callback, variable_bounds_callback)

###############################################################################
# run the simulation

time_int_tol = 1.0e-10
algorithm = TRBDF2(; autodiff = AutoFiniteDiff(), step_limiter! = limiter!)
sol = solve(ode, algorithm;
            abstol = time_int_tol, reltol = time_int_tol,
            dt = 1.0e-3, save_everystep = false,
            ode_default_options()..., callback = callbacks)
