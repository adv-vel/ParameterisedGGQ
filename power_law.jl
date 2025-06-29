using GeneralizedGauss
using LinearAlgebra
using BasisFunctions
using DataFrames
using ProgressMeter
using QuadGK
using Plots
T = Float64

function generate_rule(n, a, alpha)
    """
    Generates a Generalised Gaussian Quadrature rule for T_{n, a, alpha}
    """
    mons = [x->x^i for i in 0:n-1] # monomial basis
    mons_derivs = vcat(x->zero(x), [x->i*x^(i-1) for i in 1:n-1]) # monomial basis derivatives
    if alpha == 0
        psi_mons = [x->log(x+a)*x^i for i in 0:n-1] # log * monomial basis
        psi_mons_derivs = vcat(x->1/(x + a), [x->((x^i/(x + a) + i*log(x + a)*x^(i-1)))  for i in 1:n-1]) 
    else
        psi_mons = [x -> (x + a)^(-alpha)*x^i for i in 0:n-1] # power law * monomial basis
        psi_mons_derivs = [x -> i*x^(i-1)*(x+a)^(-alpha) - alpha*(x + a)^(-alpha-1)*x^i for i in 0:n-1] # power law * monomial basis derivatives
    end
    funs = vcat(mons, psi_mons)
    fun_derivs = vcat(mons_derivs, psi_mons_derivs)

    basis = quadbasis(funs, fun_derivs, zero(T), one(T))
    w, x = compute_gauss_rule(basis)
    return [w, x]
end

function evaluate_jac_columns(delta, x, n, alpha)
    basis_functions = zeros(T, 2*n)
    derivatives = zeros(T, 2*n)

    # handle monomials first
    basis_functions[1] = 1.0
    derivatives[1] = 0.0
    for i in 2:n 
        basis_functions[i] = x^(i-1)
        derivatives[i] = (i-1)*x^(i-2)
    end

    # handle psi monomials

    for i in 1:n
        if alpha == 0
            basis_functions[n+i] = log(x + delta)*x^(i-1)
            derivatives[n+i] = (i-1) * (x^(i-2)) * log(x + delta) + x^(i-1) / (x + delta)
        else
            basis_functions[n+i] = (x + delta)^(-alpha) * x^(i-1)
            derivatives[n+i] = (i-1)*x^(i-2)*(x+delta)^(-alpha) - alpha*(x+delta)^(-alpha-1)*x^(i-1)
        end
    end
    return basis_functions, derivatives
end

function evaluate_jacobian_inverse(delta, n, alpha)
    """
    Evaluates the Jacobian matrix for the quadrature rule at a given point x.
    """
    jacobian = zeros(T, 2*n, 2*n)
    weights, nodes = generate_rule(n, delta, alpha)

    for i in 1:n
        basis_functions, derivatives = evaluate_jac_columns(delta, nodes[i], n, alpha)
        jacobian[:, i] = basis_functions
        jacobian[:, n+i] = weights[i] * derivatives
    end
    return inv(jacobian)
end

function evaluate_partial_derivatives(delta, n, alpha)
    weights, nodes = generate_rule(n, delta, alpha)
    s = alpha + 1
    partials = zeros(T, 2*n)
    for r in 1:n
        partials[n+r] = 0
        for i in 1:n
            partials[n+r] -= weights[i] * (nodes[i]^(r-1)) / ((nodes[i] + delta)^s)
        end
        partials[n+r] += quadgk((x) -> (x^(r-1)) / ((x + delta)^s), 0, 1)[1]  # subtract the integral
        if alpha != 0
            partials[n+r] *= alpha
        end
    end
    return evaluate_jacobian_inverse(delta, n, alpha) * partials
end


function partial_derivatives_nzero(alpha)
    """
    Computes and plots partial derivatives for first element of
    GGQ rule with fixed alpha > 0 for deltas close to zero. 
    Outputs figure with line of best fit.
    """
    n = 3
    delta_range = 10 .^ range(-10, -8, length=100)  # Log-spaced delta values from 1e-12 to ~1
    partial_results = zeros(length(delta_range))

    p = Progress(length(delta_range), desc="Computing partial derivatives: ")
    for (i, delta) in enumerate(delta_range)
        partial_results[i] = abs(evaluate_partial_derivatives(delta, n, alpha)[1])
        next!(p)
    end

    # Create a loglog plot
    plt = plot(delta_range, partial_results,
        xscale=:log10, yscale=:log10,
        xlabel="δ", ylabel="|∂q₁/∂δ|",
        title="α=$alpha",
        linewidth=2, marker=:circle, markersize=3,
        color=:black,
        label="Partial Derivative wrt δ",
        dpi=500)
        # Add a line of best fit 
    if alpha != 0
        log_delta = log10.(delta_range)
        log_partials = log10.(partial_results)
        coeffs = hcat(ones(length(log_delta)), log_delta) \ log_partials
        fit_line = 10 .^ (coeffs[1] .+ coeffs[2] .* log_delta)
        plot!(plt, delta_range, fit_line, linewidth=2,
        linestyle=:dash,
        label="Fitted O(x^$(round(coeffs[2], digits=2)))", 
        color=:red)
    else
        log_delta = -1 .* log10.(delta_range)
        design_matrix = hcat(ones(length(log_delta)), log_delta)
        coeffs = design_matrix \ partial_results
        log_fit_scaled = coeffs[1] .+ coeffs[2] .* log_delta
        plot!(plt, delta_range, log_fit_scaled, linewidth=2, linestyle=:dash, label="Fitted O(-log(x))",
        color=:red)
    end

    savefig(plt, "partial_derivatives_nzero_$alpha.png")
    display(plt)
    
end

partial_derivatives_nzero(0.75)
partial_derivatives_nzero(0.5)
partial_derivatives_nzero(0.25)
partial_derivatives_nzero(0)


function partial_derivatives(alpha)
    """
    Computes and plots partial derivatives for first element of
    GGQ rule with fixed alpha > 0 for deltas close to zero. 
    """
    n = 3

    delta_range = 10 .^ range(-5, 0, length=100)
    partial_results = zeros(length(delta_range))

    p = Progress(length(delta_range), desc="Computing partial derivatives: ")
    for (i, delta) in enumerate(delta_range)
        partial_results[i] = abs(evaluate_partial_derivatives(delta, n, alpha)[1])
        next!(p)
    end
    return partial_results
end

a1 = partial_derivatives(0.75)
a2 = partial_derivatives(0.5)
a3 = partial_derivatives(0.25)
a4 = partial_derivatives(0)

delta_range = 10 .^ range(-5, 0, length=100)
plt = plot(delta_range, a4,
    yscale=:log10,
    xscale=:log10,
    xlabel="δ", ylabel="|∂q₁/∂δ|",
    linewidth=2, marker=:circle, markersize=3,
    color=:black,
    label="α = 0",
    dpi=500,
    size=(700, 400) 
)

plot!(delta_range, a3,
    yscale=:log10,
    xscale=:log10,
    xlabel="δ", ylabel="|∂q₁/∂δ|",
    linewidth=2, marker=:utriangle, markersize=3,
    color=:red,
    label="α = 0.25",
    dpi=500)

plot!(delta_range, a2,
    yscale=:log10,
    xscale=:log10,
    xlabel="δ", ylabel="|∂q₁/∂δ|",
    linewidth=2, marker=:rect, markersize=3,
    color=:blue,
    label="α = 0.5",
    dpi=500)

plot!(delta_range, a1,
    yscale=:log10,
    xscale=:log10,
    xlabel="δ", ylabel="|∂q₁/∂δ|",
    linewidth=2, marker=:circle, markersize=3,
    color=:green,
    label="α = 0.75",
    dpi=500)

savefig(plt, "partial_derivatives.png")
display(plt)
