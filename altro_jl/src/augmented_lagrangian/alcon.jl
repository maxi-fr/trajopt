const ALCONSTRAINT_PARAMS = Set((
    :use_conic_cost,
    :penalty_initial,
    :penalty_scaling,
    :penalty_max,
    :dual_max
))

Base.@kwdef mutable struct ConstraintOptions{T}
    solveropts::SolverOptions{T} = SolverOptions{T}()
    use_conic_cost::Bool = false
    penalty_initial::T = 1.0
    penalty_scaling::T = 10.0
    penalty_max::T = 1e8
    dual_max::T = 1e8
    usedefault::Dict{Symbol,Bool} = Dict(Pair.(
        ALCONSTRAINT_PARAMS,
        trues(length(ALCONSTRAINT_PARAMS))
    ))
end

function setparams!(conopts::ConstraintOptions; kwargs...)
    for (key,val) in pairs(kwargs)
        if key in ALCONSTRAINT_PARAMS
            setfield!(conopts, key, val)
            conopts.usedefault[key] = false
        end
    end
end

@generated function reset!(conopts::ConstraintOptions)
    # Use generated expression to avoid allocations since the the fields have different types
    optchecks = map(collect(ALCONSTRAINT_PARAMS)) do rawparam
        param = Expr(:quote,rawparam)
        :(conopts.usedefault[$param] && setfield!(conopts, $param, getfield(opts, $param)))
    end
    quote
        opts = conopts.solveropts
        $(Expr(:block, optchecks...))
        return nothing
    end
end

"""
    ALConstraint

A constraint on the optimization problem, which only applies to a single
knot point. Stores all of the data needed to evaluate the constraint,
as well as the Augmented Lagrangian penalty cost and its derivatives.
The default Augmented Lagrangian cost has the following form:

```math
\\frac{1}{2} \\left( || \\Pi_K(\\lambda - I_\\mu c(x) ) ||^2 - || \\lambda ||^2 \\right)
```

for a constraint in cone ``K``. To use the traditional form for equality and
inequality constraints, set `opts.use_conic_cost = false` (default).

This type also provides methods for applying the dual and penalty updates.

## Constructor
A new `ALConstraint` is created using

    ALConstraint{T}(n, m, con, inds; [sig, diffmethod, kwargs...])

with arguments:
- `n` the size of the state vector
- `m` the size of the control vector
- `con` a TrajectoryOptimization.StageConstraint
- `inds` the knot point indices at which the constraint is applied
- `kwargs` keyword arguments passed to the constructor of the [`ConstraintOptions`](@ref).

## Methods
The following methods can be used with an `ALConstraint` object:
- [`evaluate_constraint!(alcon, Z)`](@ref)
- [`constraint_jacobian!(alcon, Z)`](@ref)
- [`alcost(alcon, Z)`](@ref)
- [`algrad!(alcon, Z)`](@ref)
- [`alhess!(alcon, Z)`](@ref)
- [`dualupdate!(alcon)`](@ref)
- [`penaltyupdate!(alcon)`](@ref)
- [`normviolation!(alcon)`](@ref)
- [`max_violation(alcon)`](@ref)
- [`max_penalty(alcon)`](@ref)
- [`reset_duals!(alcon)`](@ref)
- [`reset_penalties!(alcon)`](@ref)
"""

struct ALConstraint{T, C<:TO.StageConstraint, R<:SampledTrajectory}
    nx::Vector{Int}  # state dimension     # TODO: remove these?
    nu::Vector{Int}  # control dimension
    con::C
    sig::FunctionSignature
    diffmethod::DiffMethod
    inds::Vector{Int}              # knot point indices for constraint
    vals::Vector{Vector{T}}        # constraint values
    jac::Vector{Matrix{T}}         # constraint Jacobian
    jac_scaled::Vector{Matrix{T}}  # penalty-scaled constraint Jacobian
    λ::Vector{Vector{T}}           # dual variables
    μ::Vector{Vector{T}}           # penalties
    μinv::Vector{Vector{T}}        # inverted penalties
    λbar::Vector{Vector{T}}        # approximate dual variable
    λproj::Vector{Vector{T}}       # projected dual variables
    λscaled::Vector{Vector{T}}     # scaled projected dual variables
    viol::Vector{Vector{T}}        # constraint violations
    c_max::Vector{T}

    ∇proj::Vector{Matrix{T}}   # Jacobian of projection
    ∇proj_scaled::Vector{Matrix{T}}
    ∇²proj::Vector{Matrix{T}}  # Second-order derivative of projection
    cost::Vector{T}            # (N,) vector of costs (aliased to the one in ALObjective)
    grad::Vector{Vector{T}}    # gradient of Augmented Lagrangian
    hess::Vector{Matrix{T}}    # Hessian of Augmented Lagrangian
    tmp_jac::Matrix{T}

    # Hack to avoid allocations
    # Store references to the trajectory and cost expansion
    # The trajectory is stored in a vector so it can be mutated
    # NOTE: tried to use Refs but it ended up allocating
    Z::Vector{R}
    E::CostExpansion{T}

    opts::ConstraintOptions{T}
    function ALConstraint{T}(Z::R, con::TO.StageConstraint,
                             inds::AbstractVector{<:Integer},
                             costs::Vector{T},
                             E=CostExpansion{T}(RD.dims(Z)[1:2]...);
			                 sig::FunctionSignature=StaticReturn(),
                             diffmethod::DiffMethod=UserDefined(),
                             kwargs...
    ) where {T,R<:SampledTrajectory}
        opts = ConstraintOptions{T}(;kwargs...)

        nx,nu = RD.dims(Z)
        p = RD.output_dim(con)
        P = length(inds)
        w = RD.input_dim(con)
        nz = nx + nu

        vals = [zeros(T, p) for k in inds]
        jac = [zeros(T, p, w) for k in inds]
        jac_scaled = [zeros(T, p, w) for k in inds]
        λ = [zeros(T, p) for k in inds]
        μ = [fill(opts.penalty_initial, p) for k in inds]
        μinv = [inv.(μi) for μi in μ]
        λbar = [zeros(T, p) for k in inds]
        λproj = [zeros(T, p) for k in inds]
        λscaled = [zeros(T, p) for k in inds]
        viol = [zeros(T, p) for k in inds]
        c_max = zeros(T, length(Z))

        ∇proj = [zeros(T, p, p) for k in inds]
        ∇proj_scaled = [zeros(T, p, p) for k in inds]
        ∇²proj = [zeros(T, p, p) for k in inds]

        grad = [zeros(T, nz[k]) for k in inds]
        hess = [zeros(T, nz[k], nz[k]) for k in inds]

        tmp_jac = zeros(T, p, w)

        new{T, typeof(con), R}(
            nx[inds], nu[inds], con, sig, diffmethod, inds, vals, jac, jac_scaled, λ, μ, μinv, λbar,
            λproj, λscaled, viol, c_max, ∇proj, ∇proj_scaled, ∇²proj, costs, grad, hess, tmp_jac,
            [Z], E, opts
        )
    end
end

settraj!(alcon::ALConstraint, Z::SampledTrajectory) = alcon.Z[1] = Z
setparams!(alcon::ALConstraint; kwargs...) = setparams!(alcon.opts; kwargs...)
resetparams!(alcon::ALConstraint) = reset!(alcon.opts)
RD.vectype(alcon::ALConstraint) = RD.vectype(eltype(alcon.Z[1]))

"""
    getinputinds(alcon)

Get the indices of the input to the constraint, determined by the output of
`RobotDynamics.functioninputs`. Returns `1:n+m` for generic stage constraints,
`1:n` for state constraints, and `n+1:n+m` for control constraints.
"""
function getinputinds(alcon::ALConstraint, i::Integer)
    n,m = alcon.nx[i], alcon.nu[i]
    inputtype = RD.functioninputs(alcon.con)
    if inputtype == RD.StateOnly()
        inds = 1:n
    elseif inputtype == RD.ControlOnly()
        inds = n .+ (1:m)
    else
        inds = 1:n+m
    end
end

"""
    getgrad(alcon, i)

Get the view of the entire cost gradient corresponding to the inputs for the constraint.
"""
function getgrad(alcon::ALConstraint, i::Integer)
    inds = getinputinds(alcon, i)
    return view(alcon.grad[i], inds)
end

"""
    gethess(alcon, i)

Get the view of the entire cost Hessian corresponding to the inputs for the constraint.
"""
function gethess(alcon::ALConstraint, i::Integer)
    inds = getinputinds(alcon,i)
    return view(alcon.hess[i], inds, inds)
end

"""
    evaluate_constraint!(alcon, Z)

Evaluate the constraint at all time steps, storing the result in the [`ALConstraint`](@ref).
"""
function TO.evaluate_constraints!(alcon::ALConstraint)
    TO.evaluate_constraints!(alcon.sig, alcon.con, alcon.vals, alcon.Z[1], alcon.inds)
end

"""
    constraint_jacobians!(alcon, Z)

Evaluate the constraint Jacobian at all time steps, storing the result in the
[`ALConstraint`](@ref).
"""
function TO.constraint_jacobians!(alcon::ALConstraint)
    Z = alcon.Z[1]
    sig = function_signature(alcon)
    diff = alcon.diffmethod
    for (i,k) in enumerate(alcon.inds)
        RD.jacobian!(sig, diff, alcon.con, alcon.jac[i], alcon.vals[i], Z[k])
    end
end


@doc raw"""
    alcost(alcon)

Calculates the additional cost added by the augmented Lagrangian:

```math
\sum_{i=1}^{P} \frac{1}{2 \mu_i} || \Pi_K(\lambda_i - \mu_i c(x_k)) ||^2 - || \lambda_i ||^2
```

where ``k`` is the ``i``th knot point of ``P`` to which the constraint applies, and ``K`` is the
cone for the constraint.

Assumes that the constraints have already been evaluated via [`evaluate_constraint!`](@ref).
"""
function alcost(alcon::ALConstraint{T}) where T
    use_conic = alcon.opts.use_conic_cost
    cone = TO.sense(alcon.con)
    for (i,k) in enumerate(alcon.inds)
        if use_conic
            # Use generic conic cost
            J = alcost(alcon, i)
        else
            # Special-case on the cone
            J = alcost(cone, alcon, i)
        end
        # Add to the vector of AL penalty costs stored in the ALObjective
        # Hack to avoid allocation
        alcon.cost[k] += J
    end
    return nothing
end

"""
    algrad!(alcon, Z)

Evaluate the gradient of the augmented Lagrangian penalty cost for all time
steps.

Assumes that that the constraint and constraint Jacobians have already been
evaluated via [`evaluate_constraint!`](@ref) and [`constraint_jacobian!`](@ref),
and that [`alcost`](@ref) has already been called, which evaluates the
constraint values and approximate, projected, and scaled projected multipliers.

The gradient for the generic conic cost is of the following form:

```math
    \\nabla c(x)^T \\nabla \\Pi_K(\\bar{\\lambda})^T I_{\\mu}^{-1} \\Pi_K(\\bar{\\lambda})
```

where ``\\bar{\\lambda} = \\lambda - I_{\\mu} c(x)`` are the approximate dual
variables.
"""
function algrad!(alcon::ALConstraint{T}) where T
    use_conic = alcon.opts.use_conic_cost
    cone = TO.sense(alcon.con)
    for i in eachindex(alcon.inds)
        if use_conic
            # Use generic conic cost
            algrad!(alcon, i)
        else
            # Special-case on the cone
            algrad!(cone, alcon, i)
        end
    end
end

"""
    alhess!(alcon, Z)

Evaluate the Gauss-Newton Hessian of the augmented Lagrangian penalty cost for all time
steps.

Assumes that [`alcost`](@ref) and [`algrad`](@ref) have already been called, which evaluate
the constraint values, constraint Jacobians, and approximate, projected, and scaled
projected multipliers.

The Gauss-Newton Hessian for the generic conic cost is of the following form:

```math
    \\nabla c(x)^T \\nabla \\Pi_K(\\bar{\\lambda})^T I_{\\mu}^{-1} \\nabla \\Pi_K(\\bar{\\lambda}) \\nabla c(x)
    +  \\nabla c(x)^T \\nabla^2 \\Pi_K(\\bar{\\lambda}, I_{\\mu}^{-1} \\Pi_K(\\bar{\\lambda})) \\nabla c(x)
```

where ``\\bar{\\lambda} = \\lambda - I_{\\mu} c(x)`` are the approximate dual
variables.
"""
function alhess!(alcon::ALConstraint{T}) where T
    # Assumes Jacobians have already been computed
    use_conic = alcon.opts.use_conic_cost
    cone = TO.sense(alcon.con)
    for i in eachindex(alcon.inds)
        if use_conic
            # Use generic conic cost
            alhess!(alcon, i)
        else
            # Special-case on the cone
            alhess!(cone, alcon, i)
        end
    end
end

"""
    add_alcost_expansion!(alcon, E)

Add the precomputed gradient and Hessian of the AL penalty cost to the
cost expansion stored in `E`. Assumes [`alcost(alcon)`](@ref), [`algrad!(alcon)`](@ref),
and [`alhess!(alcon)`](@ref) have already been called to evaluate these terms
about the current trajectory.
"""
function add_alcost_expansion!(alcon::ALConstraint)
    E = alcon.E  # this is aliased to the one in the iLQR solver
    for (i,k) in enumerate(alcon.inds)
        E[k].grad .+= alcon.grad[i]
        E[k].hess .+= alcon.hess[i]
    end
end


##############################
# Equality Constraints
##############################
function alcost(::TO.Equality, alcon::ALConstraint, i::Integer)
    λ, μ, c = alcon.λ[i], alcon.μ[i], alcon.vals[i]
    Iμ = Diagonal(μ)
    return λ'c + 0.5 * dot(c,Iμ,c)
end

function algrad!(::TO.Equality, alcon::ALConstraint, i::Integer)
    λbar = alcon.λbar[i]
    λ, μ, c = alcon.λ[i], alcon.μ[i], alcon.vals[i]
    ∇c = alcon.jac[i]
    grad = getgrad(alcon, i)
    # grad = alcon.grad[i]

    λbar .= λ .+ μ .* c
    matmul!(grad, ∇c', λbar)
    return nothing
end

function alhess!(::TO.Equality, alcon::ALConstraint, i::Integer)
    ∇c = alcon.jac[i]
    hess = gethess(alcon, i)
    # hess = alcon.hess[i]
    tmp = alcon.tmp_jac
    # Iμ = Diagonal(alcon.μ[i])
    # matmul!(tmp, Iμ, ∇c)
    μ = alcon.μ[i]
    for i = 1:size(∇c,1), j = 1:size(∇c,2)
        tmp[i,j] = μ[i] * ∇c[i,j]
    end
    matmul!(hess, ∇c', tmp)
    return nothing
end

##############################
# Inequality Constraints
##############################
function alcost(::TO.Inequality, alcon::ALConstraint, i::Integer)
    λ, μ, c = alcon.λ[i], alcon.μ[i], alcon.vals[i]
    a = alcon.λbar[i]
    for i = 1:length(a)
        isactive = (c[i] >= 0) | (λ[i] > 0)
        a[i] = isactive * μ[i]
    end
    Iμ = Diagonal(a)
    return λ'c + 0.5 * dot(c,Iμ,c)
end

function algrad!(::TO.Inequality, alcon::ALConstraint, i::Integer)
    ∇c, λbar = alcon.jac[i], alcon.λbar[i]
    λ, μ, c = alcon.λ[i], alcon.μ[i], alcon.vals[i]
    grad = getgrad(alcon, i)
    # grad = alcon.grad[i]
    a = alcon.λbar[i]
    for i = 1:length(a)
        isactive = (c[i] >= 0) | (λ[i] > 0)
        a[i] = isactive * μ[i]
    end
    λbar .= λ .+ a .* c
    matmul!(grad, ∇c', λbar)
    return nothing
end

function alhess!(::TO.Inequality, alcon::ALConstraint, i::Integer)
    ∇c = alcon.jac[i]
    c = alcon.vals[i]
    λ, μ = alcon.λ[i], alcon.μ[i]
    hess = gethess(alcon, i)
    # hess = alcon.hess[i]
    tmp = alcon.tmp_jac
    a = alcon.λbar[i]
    for i = 1:length(a)
        isactive = (c[i] >= 0) | (λ[i] > 0)
        a[i] = isactive * μ[i]
    end
    # Iμ = Diagonal(a)
    for i = 1:size(∇c,1), j = 1:size(∇c,2)
        tmp[i,j] = a[i] * ∇c[i,j]
    end

    # matmul!(tmp, Iμ, ∇c)
    matmul!(hess, ∇c', tmp)
    return nothing
end

@inline alcost(::TO.ConstraintSense, alcon::ALConstraint, i::Integer) = alcost(alcon, i)
@inline algrad!(::TO.ConstraintSense, alcon::ALConstraint, i::Integer) = algrad!(alcon, i)
@inline alhess!(::TO.ConstraintSense, alcon::ALConstraint, i::Integer) = alhess!(alcon, i)

##############################
# Generic Cones
##############################

function alcost(alcon::ALConstraint, i::Integer)
    dualcone = TO.dualcone(TO.sense(alcon.con))

    λ, λbar, λp, λs = alcon.λ[i], alcon.λbar[i], alcon.λproj[i], alcon.λscaled[i]
    μ, μinv, c = alcon.μ[i], alcon.μinv[i], alcon.vals[i]

    # Approximate dual
    λbar .= λ .- μ .* c

    # Projected approximate dual
    TO.projection!(dualcone, λp, λbar)

    # Scaled dual
    μinv .= inv.(μ)
    λs .= μinv .* λp

    # Cost
    Iμ = Diagonal(μinv)
    return 0.5 * (λp'λs - λ'Iμ*λ)
end

function algrad!(alcon::ALConstraint, i::Integer)
    dualcone = TO.dualcone(TO.sense(alcon.con))

    # Assume λbar and λp have already been calculated
    λbar, λp, λs = alcon.λbar[i], alcon.λproj[i], alcon.λscaled[i]
    μ, c, ∇c, Iμ∇c = alcon.μ[i], alcon.vals[i], alcon.jac[i], alcon.jac_scaled[i]
    ∇proj = alcon.∇proj[i]
    grad = getgrad(alcon, i)
    # grad = alcon.grad[i]
    tmp = alcon.tmp_jac

    # Scale the Jacobian
    p,nm = size(∇c)
    for i = 1:p
        for j = 1:nm
            Iμ∇c[i,j] = ∇c[i,j] * μ[i]
        end
    end

    # grad = -∇c'∇proj'Iμ*λp
    TO.∇projection!(dualcone, ∇proj, λbar)
    matmul!(tmp, ∇proj, Iμ∇c)  # derivative of λp wrt x
    tmp .*= -1
    matmul!(grad, tmp', λs)
    return nothing
end

function alhess!(alcon::ALConstraint, i::Integer)
    dualcone = TO.dualcone(TO.sense(alcon.con))

    λbar, λs = alcon.λbar[i], alcon.λscaled[i]
    μ, c, ∇c = alcon.μ[i], alcon.vals[i], alcon.jac_scaled[i]
    ∇proj, ∇²proj = alcon.∇proj[i], alcon.∇²proj[i]
    Iμ∇proj = alcon.∇proj_scaled[i]
    hess = gethess(alcon, i)
    # hess = alcon.hess[i]
    tmp = alcon.tmp_jac

    # Assume 𝝯proj is already computed
    # TODO: reuse this from before
    matmul!(tmp, ∇proj, ∇c)  # derivative of λp wrt x
    tmp .*= -1

    # Scale projection Jacobian
    p = length(λbar)
    μinv = alcon.μinv[i]
    for i = 1:p, j = 1:p
        Iμ∇proj[i,j] = ∇proj[i,j] * μinv[i]
    end

    # Calculate second-order projection term
    TO.∇²projection!(dualcone, ∇²proj, λbar, λs)
    matmul!(∇²proj, ∇proj', Iμ∇proj, 1.0, 1.0)

    # hess = ∇c'Iμ*(∇²proj(λs) + ∇proj'Iμ\∇proj)*Iμ*∇c
    matmul!(tmp, ∇²proj, ∇c)
    matmul!(hess, ∇c', tmp)
    return nothing
end

##############################
# Dual and Penalty Updates
##############################

"""
    dualupdate!(alcon)

Update the dual variables for all time times for a single constraint.
For now, always performs the update.
"""
function dualupdate!(alcon::ALConstraint)
    cone = TO.sense(alcon.con)
    use_conic = alcon.opts.use_conic_cost
    λ_max = alcon.opts.dual_max
    for i in eachindex(alcon.inds)
        λ, μ, c = alcon.λ[i], alcon.μ[i], alcon.vals[i]
        if use_conic
            dualupdate!(alcon, i)
        else
            # Special-case to the cone
            dualupdate!(cone, alcon, i)
        end
        # Saturate dual variables
        clamp!(λ, -λ_max, λ_max)
    end
end

function dualupdate!(::TO.Equality, alcon::ALConstraint, i::Integer)
    λbar, λ, μ, c = alcon.λbar[i], alcon.λ[i], alcon.μ[i], alcon.vals[i]
    λbar .= λ .+ μ .* c
    λ .= λbar
    return nothing
end

function dualupdate!(::TO.Inequality, alcon::ALConstraint, i::Integer)
    λbar, λ, μ, c = alcon.λbar[i], alcon.λ[i], alcon.μ[i], alcon.vals[i]
    λbar .= λ .+ μ .* c
    λ .= max.(0, λbar)
    return nothing
end

@inline dualupdate!(::TO.SecondOrderCone, alcon::ALConstraint, i::Integer) =
    dualupdate!(alcon, i)

function dualupdate!(alcon::ALConstraint, i::Integer)
    # TODO: just copy the projected duals already stored
    dualcone = TO.dualcone(TO.sense(alcon.con))
    λbar, λ, μ, c = alcon.λbar[i], alcon.λ[i], alcon.μ[i], alcon.vals[i]
    λbar .= λ .- μ .* c
    TO.projection!(dualcone, λ, λbar)
    return nothing
end

"""
    penaltyupate!(alcon)

Update the penalty terms by the geometric factor `opts.penalty_scaling`.
Always updates the penalties and updates all penalties equally, thresholding at the maximum
specified by `opts.penalty_max`.
"""
function penaltyupdate!(alcon::ALConstraint)
    μ = alcon.μ
    ϕ = alcon.opts.penalty_scaling
    μ_max = alcon.opts.penalty_max
    for i = 1:length(alcon.inds)
        μ[i] .*= ϕ
        clamp!(alcon.μ[i], 0, μ_max)
        alcon.μinv[i] .= inv.(μ[i])
    end
end

##############################
# Max Violation and Penalty
##############################

"""
    normviolation!(alcon, p)

Evaluate the `p`-norm of the constraint violations. The violation is
defined to be

```math
\\Pi_K(c(x)) - c(x)
```
These values for each time step are stored in `alcon.viol`.
"""
function normviolation!(alcon::ALConstraint, p=2, c_max=alcon.c_max)
    cone = TO.sense(alcon.con)
    c_max === alcon.c_max && (c_max .= 0)
    for (i,k) in enumerate(alcon.inds)
        TO.projection!(cone, alcon.viol[i], alcon.vals[i])
        alcon.viol[i] .-= alcon.vals[i]
        v = norm(alcon.viol[i], p)
        if isinf(p)
            c_max[k] = max(c_max[k], v)
        else
            c_max[k] = norm(SA[c_max[k], v], p)
        end
    end
    # return norm(alcon.c_max, p)
    return nothing
end
max_violation!(alcon::ALConstraint, c_max=alcon.c_max) = normviolation!(alcon, Inf, c_max)

"""
    max_penalty(alcon)

Find the maximum penalty value over all time steps.
"""
function max_penalty(alcon::ALConstraint{T}) where T
    μ_max = zero(T)
    for i = 1:length(alcon.inds)
        μ_max = max(maximum(alcon.μ[i]), μ_max)
    end
    return μ_max
end

##############################
# Reset functions
##############################
"""
    reset_duals!(alcon)

Reset the dual variables to zero.
"""
function reset_duals!(alcon::ALConstraint)
    for i = 1:length(alcon.inds)
        alcon.λ[i] .= 0
    end
end

"""
    reset_penalties!(alcon)

Reset tne penalties to the initial penalty specified by `opts.penalty_initial`.
"""
function reset_penalties!(alcon::ALConstraint)
    μ_initial = alcon.opts.penalty_initial
    for i = 1:length(alcon.inds)
        alcon.μ[i] .= μ_initial
    end
end
