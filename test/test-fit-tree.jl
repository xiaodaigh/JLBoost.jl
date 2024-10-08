using JLBoost: _fit_tree!, LogitLogLoss, depth_wise, get_leaf_nodes, keeprow_vec, max_depth_stopping_criterion

using RDatasets, DataFrames

tbl = dataset("datasets", "iris")

tbl[!, :is_setosa] = tbl.Species .== "setosa"

target = :is_setosa
features = setdiff(Symbol.(names(tbl)), [:is_setosa, :Species])

warm_start = fill(0.0, nrow(tbl))

jlt = JLBoostTree(0.0)
tree_growth = depth_wise
stopping_criterion = max_depth_stopping_criterion(6)
