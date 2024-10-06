module JLBoost

#using DataFrames
using SortingLab
using StatsBase: sample
using Base.Iterators: drop
using LossFunctions: LogitMarginLoss, deriv, deriv2, SupervisedLoss
#using Zygote: gradient, hessian
#using ForwardDiff: gradient, hessian
#using Flux: logitcrossentropy, logitbinarycrossentropy
#using RCall

export jlboost, find_best_split, _find_best_split, predict, fit_tree, logloss, jlboost!
export JLBoostTree, show, *, print, println
export LogitLogloss, deriv, deriv2, trees
export JLBoostTreeModel, JLBoostTree, WeightedJLBoostTree, features, feature_importance, vcat
export getproperty, AbstractJLBoostTree, predict
export max_depth_stopping_criterion, max_leaves_stopping_criterion


include("JLBoostTrees/JLBoostTrees.jl");
using ..JLBoostTrees: JLBoostTree, AbstractJLBoostTree, WeightedJLBoostTree,
    JLBoostTreeModel, trees, vcat, getproperty, get_leaf_nodes, treedepth, keeprow_vec,
    is_left_child, is_right_child

include("tree-growth.jl")

include("column_sampling_strategies.jl")
include("row_sampling_strategies.jl")

include("diagnostics.jl")
include("g_h.jl")
include("find_best_split.jl")

include("fit_tree_stopping_criterion.jl")
using ..TreeFitStoppingCriterion: max_depth_stopping_criterion, max_leaves_stopping_criterion


include("fit_tree.jl")


include("predict.jl")
include("save.jl")
include("jlboost-fit.jl")
include("get-features.jl")
include("feature-importance.jl")
include("tables.jl")

end # module
