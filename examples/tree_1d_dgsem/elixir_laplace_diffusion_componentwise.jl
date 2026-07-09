using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of a component-wise diffusion equation

if !isdefined(@__MODULE__, :LinearZeroAdvectionEquation1D)
    struct LinearZeroAdvectionEquation1D{RealT <: Real} <: Trixi.AbstractEquations{1, 3}
        zero_speed::RealT
    end
end

LinearZeroAdvectionEquation1D() = LinearZeroAdvectionEquation1D(0.0)

Base.similar(equations::LinearZeroAdvectionEquation1D,
             ::Type{NewRealT}) where {NewRealT} =
    LinearZeroAdvectionEquation1D(convert(NewRealT, equations.zero_speed))

Trixi.varnames(::typeof(cons2cons), ::LinearZeroAdvectionEquation1D) = ("u1", "u2", "u3")
Trixi.varnames(::typeof(cons2prim), ::LinearZeroAdvectionEquation1D) = ("u1", "u2", "u3")
Trixi.varnames(::typeof(cons2entropy), ::LinearZeroAdvectionEquation1D) = ("u1", "u2",
                                                                           "u3")

@inline Trixi.flux(u, orientation::Integer, ::LinearZeroAdvectionEquation1D) = zero(u)

@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                           equations::LinearZeroAdvectionEquation1D)
    return abs(equations.zero_speed)
end

@inline Trixi.have_constant_speed(::LinearZeroAdvectionEquation1D) = Trixi.True()
@inline Trixi.max_abs_speeds(equations::LinearZeroAdvectionEquation1D) =
    SVector(abs(equations.zero_speed))

@inline Trixi.cons2prim(u, ::LinearZeroAdvectionEquation1D) = u
@inline Trixi.prim2cons(u, ::LinearZeroAdvectionEquation1D) = u
@inline Trixi.cons2entropy(u, ::LinearZeroAdvectionEquation1D) = u
@inline Trixi.entropy(u, ::LinearZeroAdvectionEquation1D) = 0.5f0 * sum(abs2, u)

equations = LinearZeroAdvectionEquation1D()
diffusivity() = SVector(0.1, 0.0, 0.0)
equations_parabolic = LaplaceDiffusionComponentwise1D(diffusivity(), equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_central)
solver_parabolic = ParabolicFormulationLocalDG()

mesh = TreeMesh(0.0, 2.0,
                initial_refinement_level = 4,
                n_cells_max = 30_000,
                periodicity = true)

function initial_condition_laplace_diffusion_componentwise(x, t, equations)
    kappa_1 = diffusivity()[1]
    return SVector(sinpi(x[1]) * exp(-kappa_1 * pi^2 * t),
                   cospi(2 * x[1]),
                   1 + 0.25 * sinpi(3 * x[1]))
end
initial_condition = initial_condition_laplace_diffusion_componentwise

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = solver_parabolic,
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 100)
alive_callback = AliveCallback(analysis_interval = 100)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL35();
            dt = 1.0e-4, adaptive = false,
            ode_default_options()..., callback = callbacks)
