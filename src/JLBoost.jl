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
export getproperty, AbstractJLBoostTree, predict

# include("traitwrappers.jl")

include("JLBoostTree/JLBoostTree.jl");
using ..JLBoostTrees: JLBoostTree, AbstractJLBoostTree, WeightedJLBoostTree,
    JLBoostTreeModel, trees, vcat, getproperty, get_leaf_nodes

include("column_sampling_strategies.jl")
include("row_sampling_strategies.jl")

include("diagnostics.jl")
include("g_h.jl")
include("best_split.jl")
include("tree-growth.jl")


inlcude("fit_tree_stopping_criterion.jl")
using ..TreeFitStoppingCriterions: max_depth
include("fit_tree.jl")

include("predict.jl")
include("save.jl")
include("jlboost-fit.jl")
include("get-features.jl")
include("feature-importance.jl")
include("tables.jl")

end # module
