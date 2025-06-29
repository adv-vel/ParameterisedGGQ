module interpolation_functions
# This module contains functions for rational interpolation and 
# getting training data for other interpolation methods in python.

using GeneralizedGauss
using GeneralizedGauss
using LinearAlgebra
using BasisFunctions
using DataFrames
using ProgressMeter
using QuadGK
using Plots
using CSV
using BaryRational
using Statistics
using Random


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
    return vcat(w, x)
end

function chebyshev_points(n)
    k = 0:n-1
    cheb_std = cos.(k * π / (n-1))          # Lobatto nodes on [-1,1]
    cheb_01  = (cheb_std .+ 1) / 2          # rescale
    return reverse(cheb_01)
end


function barycentric_training_data(n, alpha)
    if alpha == 0
        beta = 2
    else
        beta = 1/(1 - alpha)
    end
    xs = chebyshev_points(n).^(beta)
    rule = [generate_rule(3, x, alpha) for x in xs]
    x_trans = (xs).^(1/beta)
    df = DataFrame(ColumnA = x_trans, ColumnB = rule)
    output_file = joinpath("data", "cheb_training_data_$(n)_$(alpha).csv")
    CSV.write(output_file, df)
    return df
end

export aaa_training_data
function aaa_training_data(n, alpha)
    xs = chebyshev_points(n)
    rule = [generate_rule(3, x, alpha) for x in xs]
    df = DataFrame(ColumnA = xs, ColumnB = rule)
    output_file = joinpath("data", "aaa_training_data_$(n)_$(alpha).csv")
    CSV.write(output_file, df)
end

function uniform_training_data(n, alpha)
    xs = range(0, 1, length=n).^2
    rule = [generate_rule(3, x, alpha) for x in xs]
    df = DataFrame(ColumnA = xs, ColumnB = rule)
    output_file = joinpath("data", "uniform_training_data_$(n)_$(alpha).csv")
    CSV.write(output_file, df)
end

function spline_training_data(n, alpha)
    if alpha == 0
        beta = 3
    else
        beta = 2/(1 - alpha)
    end
    xs = range(0, 1, length=n).^(beta)
    rule = [generate_rule(3, x, alpha) for x in xs]
    x_trans = (xs).^(1/beta)
    df = DataFrame(ColumnA = x_trans, ColumnB = rule)
    output_file = joinpath("data", "spline_training_data_$(n)_$(alpha).csv")
    CSV.write(output_file, df)
end


function testing_data(alpha)
    Random.seed!(100)
    test_deltas = rand(3000) # random points in [0, 1]
    rule = [generate_rule(3, x, alpha) for x in test_deltas]
    df = DataFrame(ColumnA = test_deltas, ColumnB = rule)
    output_file = joinpath("data", "testing_data_$(alpha).csv")
    CSV.write(output_file, df)
end

function testing_data_near_zero(alpha)
    Random.seed!(100)
    test_deltas = 10 .^ range(-6, -1; length=3000)
    rule = [generate_rule(3, x, alpha) for x in test_deltas]
    df = DataFrame(ColumnA = test_deltas, ColumnB = rule)
    output_file = joinpath("data", "testing_data_$(alpha)_nzero.csv")
    CSV.write(output_file, df)
end

end