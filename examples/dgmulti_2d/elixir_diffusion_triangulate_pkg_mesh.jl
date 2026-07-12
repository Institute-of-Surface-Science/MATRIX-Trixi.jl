using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Equation and analytical solution

diffusivity = 5.0e-2
equations = LinearDiffusionEquation2D(diffusivity)

# Exact solution of the diffusion equation on [-1, 1]^2 with fixed values at
# y = +/-1 and insulating boundaries at x = +/-1.
function initial_condition_diffusion(x, t, equations)
    decay_rate = equations.diffusivity * (pi / 2)^2
    transient_part = exp(-decay_rate * t) * cospi(x[2] / 2)
    scalar = one(x[2]) + x[2] + transient_part
    return SVector(scalar)
end
initial_condition = initial_condition_diffusion

###############################################################################
# DG discretization

polydeg = 3
solver = DGMulti(polydeg = polydeg,
                 element_type = Tri(),
                 approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central),
                 volume_integral = VolumeIntegralWeakForm())
solver_parabolic = ParabolicFormulationLocalDG(1.0)

###############################################################################
# Triangular mesh with integer boundary markers

mesh_size = 0.25

# `RectangularDomain` orders segment markers as bottom, right, top, left.
# Non-contiguous values exercise the conversion from integer markers to names.
segment_markers = SVector(10, 20, 30, 40)
domain = StartUpDG.RectangularDomain(; segment_markers)
mesh_io = StartUpDG.triangulate_domain(domain; h = mesh_size)

boundary_tag_map = Dict(:bottom => 10,
                        :right => 20,
                        :top => 30,
                        :left => 40)
mesh = DGMultiMesh(solver, mesh_io, boundary_tag_map)

###############################################################################
# Boundary conditions

boundary_value_bottom(x, t, equations) = SVector(zero(x[1]))
boundary_value_top(x, t, equations) = SVector(2 * one(x[1]))
boundary_normal_flux_zero(x, t, equations) = SVector(zero(x[1]))

boundary_condition_bottom = BoundaryConditionDirichlet(boundary_value_bottom)
boundary_condition_top = BoundaryConditionDirichlet(boundary_value_top)
boundary_condition_insulating = BoundaryConditionNeumann(boundary_normal_flux_zero)

boundary_conditions = (; bottom = boundary_condition_bottom,
                       right = boundary_condition_insulating,
                       top = boundary_condition_top,
                       left = boundary_condition_insulating)

###############################################################################
# Semidiscretization

semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic,
                                   boundary_conditions)

###############################################################################
# ODE problem and callbacks

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_callback = AnalysisCallback(semi; interval = 100,
                                     uEltype = real(solver))
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# Time integration

time_int_tol = 1.0e-8
sol = solve(ode, RDPK3SpFSAL49();
            abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
