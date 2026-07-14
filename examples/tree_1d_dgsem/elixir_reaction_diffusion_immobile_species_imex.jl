using OrdinaryDiffEqSDIRK
using ADTypes
using Trixi

###############################################################################
# semidiscretization of a stiff reaction-diffusion system

# We solve the linear reversible exchange problem
#
#   d_t mobile   = D * d_xx mobile - k * mobile + p * immobile,
#   d_t immobile =                       k * mobile - p * immobile.
#
# Only `mobile` diffuses. The reaction transfers material between both variables,
# so their sum is conserved on the periodic domain.
diffusivity() = 1.0e-2
reaction_rate_forward() = 100.0
reaction_rate_backward() = 50.0

# `ZeroFluxEquations1D` provides the two-variable carrier required by
# `SemidiscretizationHyperbolicParabolic`. Its hyperbolic RHS is identically zero,
# leaving the explicit part of the split empty.
equations = ZeroFluxEquations1D(2)

# The diffusivity tuple corresponds to `(mobile, immobile)`. A zero entry suppresses
# the parabolic flux of that component without suppressing its local reaction term.
equations_parabolic = LaplaceDiffusionComponentwise1D((diffusivity(), 0.0), equations)

# The gradient argument is unused since the reversible exchange is local in space.
# The opposite signs make the reaction contribution to `mobile + immobile` vanish.
function source_terms_reaction(u, gradients, x, t, equations_parabolic)
    mobile, immobile = u
    reaction = reaction_rate_forward() * mobile - reaction_rate_backward() * immobile
    return SVector(-reaction, reaction)
end

# DGSEM discretizes the spatial operators. The minimum-dissipation LDG formulation
# is used for the diffusion term.
solver = DGSEM(polydeg = 3)
solver_parabolic = ParabolicFormulationLocalDG()

# Periodicity removes boundary exchange, which makes conservation of the total amount
# a direct check of the reaction-diffusion discretization.
mesh = TreeMesh(0.0, 2.0 * pi,
                initial_refinement_level = 4,
                n_cells_max = 10_000,
                periodicity = true)

# On a periodic domain, every Fourier mode evolves independently. For wave number `n`,
# diffusion contributes `-D * n^2` to the mobile variable and the mode amplitudes obey
#
#   d_t [mobile; immobile] = [-D*n^2-k  p; k  -p] * [mobile; immobile].
#
# This helper applies the closed-form exponential of that two-by-two matrix. It is used
# below to provide a time-dependent analytical reference for `AnalysisCallback`.
function evolve_reaction_diffusion_mode(mobile, immobile, wave_number, t;
                                        diffusion = diffusivity(),
                                        forward_rate = reaction_rate_forward(),
                                        backward_rate = reaction_rate_backward())
    diffusion_rate = diffusion * wave_number^2

    matrix_11 = -diffusion_rate - forward_rate
    matrix_22 = -backward_rate
    half_trace = 0.5 * (matrix_11 + matrix_22)
    discriminant = sqrt(0.25 * (matrix_11 - matrix_22)^2 +
                        forward_rate * backward_rate)

    cosh_discriminant_t = cosh(discriminant * t)
    sinh_discriminant_t_over_discriminant = if iszero(discriminant)
        t
    else
        sinh(discriminant * t) / discriminant
    end
    shifted_mobile = (matrix_11 - half_trace) * mobile + backward_rate * immobile
    shifted_immobile = forward_rate * mobile + (matrix_22 - half_trace) * immobile

    factor = exp(half_trace * t)
    return factor * SVector(cosh_discriminant_t * mobile +
                   sinh_discriminant_t_over_discriminant * shifted_mobile,
                   cosh_discriminant_t * immobile +
                   sinh_discriminant_t_over_discriminant * shifted_immobile)
end

# At `t = 0`, the state is
#
#   mobile   = 1 + 0.1 * sin(x),
#   immobile = 0.25 + 0.05 * cos(x).
#
# Evolving the constant, sine, and cosine amplitudes separately gives the exact solution
# for all times and turns the reported norms into actual numerical errors.
function initial_condition_reaction_diffusion(x, t, equations)
    constant_mode = evolve_reaction_diffusion_mode(1.0, 0.25, 0, t)
    sine_mode = evolve_reaction_diffusion_mode(0.1, 0.0, 1, t)
    cosine_mode = evolve_reaction_diffusion_mode(0.0, 0.05, 1, t)
    return constant_mode + sin(x[1]) * sine_mode + cos(x[1]) * cosine_mode
end

# The reaction is placed in `source_terms_parabolic` so that it belongs to
# the first component of the SplitODEProblem and is treated implicitly by IMEX solvers.
source_terms_parabolic = source_terms_reaction

# The boundary-condition tuple contains the hyperbolic and parabolic conditions,
# respectively. Both are periodic for this example.
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition_reaction_diffusion,
                                             solver;
                                             solver_parabolic = solver_parabolic,
                                             source_terms_parabolic = source_terms_parabolic,
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solver and callbacks

# The final time spans several reaction time scales, approximately `1 / (k + p)`,
# while keeping the example small enough for routine testing.
tspan = (0.0, 0.1)
# For larger stiff reaction-diffusion systems, pass `jac_prototype_parabolic`
# and optionally `colorvec_parabolic` to `semidiscretize`.
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
# Since the initial-condition function is the analytical solution for arbitrary `t`,
# the analysis callback reports errors against that solution.
analysis_callback = AnalysisCallback(semi, interval = 100)
callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

# `KenCarp4` treats the first SplitODEProblem component implicitly and the second one
# explicitly. Here this means implicit diffusion and reaction with an exactly zero
# explicit RHS. Finite-difference Jacobians are compatible with Trixi's mutable caches.
ode_alg = KenCarp4(autodiff = AutoFiniteDiff())

# No parabolic CFL callback is needed: adaptive implicit stepping resolves the stiff
# reaction and diffusion terms according to the requested tolerances.
sol = solve(ode, ode_alg;
            abstol = 1.0e-8, reltol = 1.0e-8,
            ode_default_options()..., callback = callbacks)
