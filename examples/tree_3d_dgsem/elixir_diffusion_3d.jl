using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the scalar diffusion equation

diffusivity() = 0.1
equations = LinearDiffusionEquation3D(diffusivity())

solver = DGSEM(polydeg = 3)
solver_parabolic = ParabolicFormulationLocalDG()

coordinates_min = (-convert(Float64, pi), -convert(Float64, pi),
                   -convert(Float64, pi))
coordinates_max = (convert(Float64, pi), convert(Float64, pi),
                   convert(Float64, pi))
initial_refinement_level = 2

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 100_000,
                periodicity = true)

function initial_condition_diffusion_3d(x, t, equations)
    scalar = sin(x[1]) * sin(x[2]) * sin(x[3]) * exp(-3 * equations.diffusivity * t)
    return SVector(scalar)
end

semi = SemidiscretizationParabolic(mesh, equations, initial_condition_diffusion_3d,
                                   solver;
                                   solver_parabolic = solver_parabolic,
                                   boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.05)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 100)
alive_callback = AliveCallback(analysis_interval = 100)
stepsize_callback = StepsizeCallback(cfl_parabolic = 0.05)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL35();
            dt = stepsize_callback(ode), adaptive = false,
            ode_default_options()..., callback = callbacks)
