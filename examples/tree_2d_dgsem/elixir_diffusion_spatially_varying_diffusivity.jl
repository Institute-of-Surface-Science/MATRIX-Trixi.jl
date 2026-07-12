using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Pure diffusion equation with a smooth spatially varying coefficient

amplitude() = 0.25
diffusivity_bound() = 1.25

@inline function diffusivity(x, t, equations)
    local_amplitude = convert(eltype(x), amplitude())
    return one(eltype(x)) + local_amplitude * sinpi(x[1]) * sinpi(x[2])
end

diffusivity_coefficient = SpatiallyVaryingDiffusivity(diffusivity,
                                                      diffusivity_bound())
equations = LinearDiffusionEquation2D(diffusivity_coefficient)

@inline function initial_condition(x, t, equations)
    scalar = exp(-t) * sinpi(x[1]) * sinpi(x[2])
    return SVector(scalar)
end

@inline function source_terms(u, gradients, x, t, equations)
    sx = sinpi(x[1])
    sy = sinpi(x[2])
    cx = cospi(x[1])
    cy = cospi(x[2])

    exp_t = exp(-t)
    scalar = exp_t * sx * sy
    local_amplitude = convert(eltype(x), amplitude())
    local_diffusivity = one(eltype(x)) + local_amplitude * sx * sy
    laplace_u = -2 * pi^2 * scalar
    grad_diffusivity_dot_grad_u = local_amplitude * pi^2 * exp_t *
                                  ((cx * sy)^2 + (sx * cy)^2)
    divergence = local_diffusivity * laplace_u + grad_diffusivity_dot_grad_u

    return SVector(-scalar - divergence)
end

###############################################################################
# TreeMesh and parabolic semidiscretization

polydeg = 3
solver = DGSEM(polydeg = polydeg)
solver_parabolic = ParabolicFormulationBassiRebay1()

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 3
mesh = TreeMesh(coordinates_min, coordinates_max;
                initial_refinement_level,
                n_cells_max = 100_000,
                periodicity = true)

semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic,
                                   source_terms,
                                   boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE problem and time integration

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi; interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)
stepsize_callback = StepsizeCallback(cfl_parabolic = 0.05)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

ode_algorithm = RDPK3SpFSAL35()
sol = solve(ode, ode_algorithm;
            dt = stepsize_callback(ode), adaptive = false,
            ode_default_options()..., callback = callbacks)
