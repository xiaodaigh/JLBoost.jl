using Pkg
Pkg.activate(".")
@time using DataFrames
@time using JDF
@time using JLBoost, LossFunctions


###############################################################################
# testing best_split
###############################################################################
@time a = loadjdf("c:/data/GiveMeSomeCredit/cs-training.jdf")

loss = LogitLogLoss()
features = setdiff(names(a), [:SeriousDlqin2yrs, :column1])
target = :SeriousDlqin2yrs
warmstart = fill(0, length(a.age))
verbose = false
lambda = 0
gamma = 0

using MLJ

treem = jlboost(a, target, feats)

predict(treem, a)
get_features(treem)
feature_importance(treem, a)
