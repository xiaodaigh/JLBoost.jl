module JLBoost

#using DataFrames
using SortingLab
using StatsBase: sample
using Base.Iterators: drop
using LossFunctions: LogitProbLoss, deriv, deriv2, SupervisedLoss
#using Zygote: gradient, hessian
#using ForwardDiff: gradient, hessian
#using Flux: logitcrossentropy, logitbinarycrossentropy
#using RCall

export jlboost, best_split, _best_split, predict, fit_tree, logloss, jlboost!
export JLBoostTree, show, *, print, println
export LogitLogloss, value, deriv, deriv2, trees
export JLBoostTreeModel, JLBoostTree, WeightedJLBoostTree, features, feature_importance, vcat

include("traitwrappers.jl")

include("JLBoostTree.jl");
using ..JLBoostTrees: JLBoostTree, AbstractJLBoostTree, WeightedJLBoostTree,
    JLBoostTreeModel, trees, vcat


include("diagnostics.jl")
include("g_h.jl")
include("best_split.jl")
include("fit_tree.jl")
include("predict.jl")
include("save.jl")
include("jlboost-fit.jl")
include("get-features.jl")
include("feature-importance.jl")
include("mlj.jl")
include("tables.jl")

end # module
