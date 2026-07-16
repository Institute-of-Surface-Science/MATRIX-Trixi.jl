using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of a diffusion-reaction equation with time-dependent coefficients

# We solve
#
#   d_t u = d_x(D(t) * d_x u) - k(t) * u + f(x, t)
#
# on a periodic domain. The spatial and equation arguments are intentionally retained to
# demonstrate the same provider API required by a coefficient D(u, x, t).
@inline diffusivity_value(x, t, equations) = 0.2 * (1 + 0.25 * sin(t))
@inline reaction_rate(t) = 0.3 * (1 + 0.2 * cos(t))

diffusivity = SpatiallyVaryingDiffusivity(diffusivity_value, 0.25)
equations = LinearDiffusionEquation1D(diffusivity)

# For u_exact = exp(-t) * sin(x), both d_t u_exact and d_xx u_exact equal
# `-u_exact`. The source below therefore manufactures this exact solution for arbitrary
# values of D(t) and k(t).
@inline function exact_solution(x, t, equations::LinearDiffusionEquation1D)
    return SVector(exp(-t) * sin(x[1]))
end
initial_condition = exact_solution

@inline function source_terms(u, _gradients, x, t,
                              equations::LinearDiffusionEquation1D)
    diffusion = Trixi.diffusivity_value(equations.diffusivity, x, t, equations)
    reaction = reaction_rate(t)
    u_exact = exact_solution(x, t, equations)[1]
    manufactured_forcing = (diffusion + reaction - 1) * u_exact
    return SVector(-reaction * u[1] + manufactured_forcing)
end

solver = DGSEM(polydeg = 3)
solver_parabolic = ParabolicFormulationLocalDG()

mesh = TreeMesh(-Float64(pi), Float64(pi),
                initial_refinement_level = 4,
                periodicity = true,
                n_cells_max = 30_000)

semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic = solver_parabolic,
                                   source_terms = source_terms,
                                   boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solver and callbacks

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 100)
alive_callback = AliveCallback(analysis_interval = 100)
stepsize_callback = StepsizeCallback(cfl_parabolic = 0.05)
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

# The callback re-evaluates the time-dependent diffusivity when selecting every time step.
sol = solve(ode, RDPK3SpFSAL49();
            dt = stepsize_callback(ode), adaptive = false,
            ode_default_options()..., callback = callbacks)
