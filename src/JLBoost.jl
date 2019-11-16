module JLBoost

using DataFrames
using SortingLab
using StatsBase: sample
using Base.Iterators: drop
using LossFunctions: LogitProbLoss, deriv, deriv2, SupervisedLoss
#using Zygote: gradient, hessian
#using ForwardDiff: gradient, hessian
#using Flux: logitcrossentropy, logitbinarycrossentropy
#using RCall

export jlboost, best_split, _best_split, predict, fit_tree, logloss, jlboost!
export update_weight

include("JLBoostTree.jl"); using ..JLBoostTrees: JLBoostTreeNode

include("diagnostics.jl")
include("g_h.jl")
include("best_split.jl")
include("fit_tree.jl")
include("predict.jl")
include("jlboost-fit.jl")


# update the weight once so that it starts at a better point
function update_weight(loss_fn, df, target, prev_w, lambda)
    target_vec = df[!, target];
    prev_w_vec = df[!, prev_w];

    -sum(g.(loss_fn, target_vec, prev_w_vec))/(sum(h.(loss_fn, target_vec, prev_w_vec)) + lambda)
end

"""
	apply_split(df::AbstractDataFrame, feature, split_at, lweight, rweight)

Apply split to the dataframe df
"""
function apply_split(df::AbstractDataFrame, feature, split_at, lweight, rweight)
    df[df[feature] .<= split_at,:prev_w] = df[df[feature] .<= split_at,:prev_w] .+ lweight
    df[df[feature] .> split_at,:prev_w] = df[df[feature] .> split_at,:prev_w] .+ rweight
    df
end

end # module
