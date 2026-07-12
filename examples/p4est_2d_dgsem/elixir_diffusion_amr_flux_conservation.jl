using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Pure diffusion equation and exact solution

diffusivity() = 1.0e-2
amplitude() = 1.0
equations = LinearDiffusionEquation2D(diffusivity())

# Fourier coefficients of sinpi(z)^8. Each cosine mode is evolved exactly by
# the diffusion equation, while the stationary linear profile supplies nonzero
# normal fluxes on the left and right boundaries.
const sine_power_coefficients = (35.0 / 128.0, -56.0 / 128.0, 28.0 / 128.0,
                                 -8.0 / 128.0, 1.0 / 128.0)

@inline function diffused_sine_power(z, t, diffusivity)
    result = zero(z)
    for mode in 0:4
        wave_number = 2 * mode * pi
        result += sine_power_coefficients[mode + 1] * cos(wave_number * z) *
                  exp(-diffusivity * wave_number^2 * t)
    end
    return result
end

@inline function exact_solution(x, t, equations::LinearDiffusionEquation2D)
    transient_x = diffused_sine_power(x[1], t, max_diffusivity(equations))
    transient_y = diffused_sine_power(x[2], t, max_diffusivity(equations))
    return SVector(x[1] + amplitude() * transient_x * transient_y)
end
initial_condition = exact_solution

###############################################################################
# P4est mesh and parabolic semidiscretization

polydeg = 3
solver = DGSEM(polydeg = polydeg)
solver_parabolic = ParabolicFormulationBassiRebay1()

trees_per_dimension = (2, 2)
mesh = P4estMesh(trees_per_dimension;
                 polydeg = polydeg,
                 coordinates_min = (0.0, 0.0),
                 coordinates_max = (1.0, 1.0),
                 initial_refinement_level = 0,
                 periodicity = false)
initial_ncells = Trixi.ncells(mesh)

@inline flux_x_neg(x, t, equations) = SVector(-max_diffusivity(equations))
@inline flux_x_pos(x, t, equations) = SVector(max_diffusivity(equations))
@inline flux_zero(x, t, equations) = SVector(zero(x[1]))

boundary_conditions = (; x_neg = BoundaryConditionNeumann(flux_x_neg),
                       x_pos = BoundaryConditionNeumann(flux_x_pos),
                       y_neg = BoundaryConditionNeumann(flux_zero),
                       y_pos = BoundaryConditionNeumann(flux_zero))

semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic,
                                   boundary_conditions)

###############################################################################
# ODE problem and adaptive mesh refinement

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)
initial_mass = first(integrate(ode.u0, semi; normalize = false))

summary_callback = SummaryCallback()
analysis_interval = 50
normal_flux_x_neg = AnalysisSurfaceIntegral((:x_neg,),
                                            NormalParabolicFlux(1;
                                                                name = :flux_x_neg))
normal_flux_x_pos = AnalysisSurfaceIntegral((:x_pos,),
                                            NormalParabolicFlux(1;
                                                                name = :flux_x_pos))
normal_flux_y_neg = AnalysisSurfaceIntegral((:y_neg,),
                                            NormalParabolicFlux(1;
                                                                name = :flux_y_neg))
normal_flux_y_pos = AnalysisSurfaceIntegral((:y_pos,),
                                            NormalParabolicFlux(1;
                                                                name = :flux_y_pos))
analysis_callback = AnalysisCallback(semi;
                                     interval = analysis_interval,
                                     extra_analysis_integrals = (normal_flux_x_neg,
                                                                 normal_flux_x_pos,
                                                                 normal_flux_y_neg,
                                                                 normal_flux_y_pos))
alive_callback = AliveCallback(analysis_interval = analysis_interval)

base_level = 0
med_level = 1
max_level = 3
med_threshold = 0.02
max_threshold = 0.15
amr_indicator = IndicatorLöhner(semi, variable = first)
amr_controller = ControllerThreeLevel(semi, amr_indicator;
                                      base_level,
                                      med_level, med_threshold,
                                      max_level, max_threshold)
amr_interval = 5
amr_callback = AMRCallback(semi, amr_controller;
                           interval = amr_interval,
                           adapt_initial_condition = false)

# A small parabolic CFL is used so that several accepted steps and AMR decisions
# occur during this short example.
cfl_parabolic = 2.0e-2
stepsize_callback = StepsizeCallback(; cfl_parabolic)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        amr_callback, stepsize_callback)

###############################################################################
# Time integration

ode_algorithm = CarpenterKennedy2N54(williamson_condition = false)
sol = solve(ode, ode_algorithm;
            dt = stepsize_callback(ode), adaptive = false,
            ode_default_options()..., callback = callbacks,
            maxiters = 500_000)

###############################################################################
# Post-AMR conservation and boundary-flux diagnostics

final_ncells = Trixi.ncells(mesh)
final_nmortars = Trixi.nmortars(semi.cache.mortars)
final_mass = first(integrate(sol.u[end], semi; normalize = false))

du_final_ode = similar(sol.u[end])
Trixi.rhs_parabolic!(du_final_ode, sol.u[end], semi, sol.t[end])
mass_rate = first(integrate(du_final_ode, semi; normalize = false))

u_final = Trixi.wrap_array(sol.u[end], semi)
du_final = Trixi.wrap_array(du_final_ode, semi)

boundary_fluxes = (;
                   x_neg = Trixi.analyze(normal_flux_x_neg, du_final, u_final,
                                         sol.t[end], semi),
                   x_pos = Trixi.analyze(normal_flux_x_pos, du_final, u_final,
                                         sol.t[end], semi),
                   y_neg = Trixi.analyze(normal_flux_y_neg, du_final, u_final,
                                         sol.t[end], semi),
                   y_pos = Trixi.analyze(normal_flux_y_pos, du_final, u_final,
                                         sol.t[end], semi))

expected_flux_x_neg = -diffusivity()
expected_flux_x_pos = diffusivity()
expected_flux_y_neg = 0.0
expected_flux_y_pos = 0.0
net_boundary_flux = sum(boundary_fluxes)
