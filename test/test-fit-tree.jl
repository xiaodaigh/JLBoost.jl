using JLBoost: depth_wise

using JLBoost: _fit_tree!, LogitLogLoss

using RDatasets, DataFrames

loss = LogitLogLoss()

tbl = dataset("datasets", "iris")

tbl[:is_setosa] = tbl.Species .== "setosa"

target = :is_setosa
features = setdiff(Symbol.(names(tbl)), [:is_setosa, :Species])

warm_start = fill(0.0, nrow(tbl))
lambda = 0
gamma = 0
verbose = false

jlt = JLBoostTree(0.0)
tree_growth = depth_wise
stopping_criterion = max_depth(6)

# _fit_tree!(loss, tbl, target, features, warm_start,
#     jlt::AbstractJLBoostTree = JLBoostTree(0.0),
#     tree_growth::Function = depth_wise,
#     stopping_criterion::Function = max_depth(6);
#     #colsample_bynode = 1, colsample_bylevel = 1,
# 	kwargs...)

