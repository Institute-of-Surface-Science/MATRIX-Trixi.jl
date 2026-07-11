using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of a diffusion-reaction equation with time-dependent coefficients

# We solve
#
#   d_t u = d_x(D(t) * d_x u) - k(t) * u + f(x, t)
#
# on a periodic domain. The spatial argument of `diffusivity` is intentionally retained
# to demonstrate the same API required by a coefficient D(x, t).
@inline diffusivity(x, t) = 0.2 * (1 + 0.25 * sin(t))
@inline reaction_rate(t) = 0.3 * (1 + 0.2 * cos(t))

# This example-local equation implements only the extended parabolic flux API. Existing Trixi
# equations retain the original kernel for the shorter, time-independent flux method.
struct TimeDependentDiffusionEquation1D{F} <: Trixi.AbstractLaplaceDiffusion{1, 1}
    diffusivity::F
end

Trixi.varnames(::typeof(cons2cons), ::TimeDependentDiffusionEquation1D) = ("scalar",)
Trixi.varnames(::typeof(cons2prim), ::TimeDependentDiffusionEquation1D) = ("scalar",)
Trixi.varnames(::typeof(cons2entropy), ::TimeDependentDiffusionEquation1D) = ("scalar",)

@inline Trixi.cons2prim(u, ::TimeDependentDiffusionEquation1D) = u
@inline Trixi.cons2entropy(u, ::TimeDependentDiffusionEquation1D) = u

# The coefficient changes during integration. Marking it nonconstant also prevents
# `linear_structure` from treating the semidiscretization as a fixed autonomous operator.
@inline function Trixi.have_constant_diffusivity(::TimeDependentDiffusionEquation1D)
    return Trixi.False()
end

@inline function Trixi.have_space_time_dependent_flux(::TimeDependentDiffusionEquation1D)
    return Trixi.True()
end

@inline function Trixi.flux(u, gradients, orientation::Integer, x, t,
                            equations::TimeDependentDiffusionEquation1D)
    dudx, = gradients
    return equations.diffusivity(x, t) * dudx
end

equations = TimeDependentDiffusionEquation1D(diffusivity)

# For u_exact = exp(-t) * sin(x), both d_t u_exact and d_xx u_exact equal
# `-u_exact`. The source below therefore manufactures this exact solution for arbitrary
# values of D(t) and k(t).
@inline function exact_solution(x, t, equations::TimeDependentDiffusionEquation1D)
    return SVector(exp(-t) * sin(x[1]))
end
initial_condition = exact_solution

@inline function source_terms(u, _gradients, x, t,
                              equations::TimeDependentDiffusionEquation1D)
    diffusion = equations.diffusivity(x, t)
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
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

# Adaptive error control evaluates D(t) and k(t) at every Runge-Kutta stage. A
# `StepsizeCallback` is intentionally not used since time-dependent CFL estimates are outside
# the scope of this example.
sol = solve(ode, RDPK3SpFSAL49();
            abstol = 1.0e-8, reltol = 1.0e-8,
            ode_default_options()..., callback = callbacks)
