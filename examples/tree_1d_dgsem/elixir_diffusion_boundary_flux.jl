using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Semidiscretization of the pure diffusion equation

diffusivity = 0.5
equations = LinearDiffusionEquation1D(diffusivity)

solver = DGSEM(polydeg = 3)
solver_parabolic = ParabolicFormulationLocalDG()

mesh = TreeMesh(0.0, 1.0,
                initial_refinement_level = 2,
                n_cells_max = 1_000,
                periodicity = false)

# This steady linear profile has the constant parabolic flux kappa * u_x = -0.5.
function linear_profile(x, t, equations)
    return SVector(1.0 - x[1])
end
initial_condition = linear_profile

boundary_conditions = (; x_neg = BoundaryConditionDirichlet(linear_profile),
                       x_pos = BoundaryConditionDirichlet(linear_profile))

semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic,
                                   boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.01)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# The normal is outward, so the normal parabolic flux is +0.5 on the left and
# -0.5 on the right. Their sum is zero. Multiplication by -1 converts the
# parabolic flux into the conventional Fickian flux, which is +0.5 on the right.
normal_flux_left = AnalysisSurfaceIntegral((:x_neg,),
                                           NormalParabolicFlux(1;
                                                               name = :normal_flux_left))
normal_flux_right = AnalysisSurfaceIntegral((:x_pos,),
                                            NormalParabolicFlux(1;
                                                                name = :normal_flux_right))
normal_flux_total = AnalysisSurfaceIntegral((:x_neg, :x_pos),
                                            NormalParabolicFlux(1;
                                                                name = :normal_flux_total))
fick_outflow_right = AnalysisSurfaceIntegral((:x_pos,),
                                             NormalParabolicFlux(1;
                                                                 factor = -1,
                                                                 name = :fick_outflow_right))

analysis_callback = AnalysisCallback(semi;
                                     interval = 10,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (normal_flux_left,
                                                           normal_flux_right,
                                                           normal_flux_total,
                                                           fick_outflow_right))

callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# Run the simulation

sol = solve(ode, RDPK3SpFSAL35(); dt = 1.0e-4, adaptive = false,
            ode_default_options()..., callback = callbacks)
