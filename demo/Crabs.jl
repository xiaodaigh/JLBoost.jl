using Pkg; Pkg.activate(".")
using MLJ, StatsBase, Random, PyPlot, CategoricalArrays, PrettyPrinting, DataFrames
X, y = @load_crabs
X = DataFrame(X)

using XGBoost, MLJ
@load XGBoostClassifier
xgb  = XGBoostClassifier()
xgbm = machine(xgb, X, y)
r = range(xgb, :num_round, lower=10, upper=500)
curve = learning_curve!(xgbm, resampling=CV(),
                        range=r, resolution=25,
                        measure=cross_entropy)


using JLBoost
xgb = JLBoostClassifier()
xgbm = machine(xgb, X, y)
r = range(xgb, :nrounds, lower=10, upper=500)
curve = learning_curve!(xgbm, resampling=CV(),
                        range=r, resolution=25,
                        measure=cross_entropy)


using DataAPI, CategoricalArrays

y
DataAPI.refarray(y)
DataAPI.refvalue(y, y[1])

DataAPI.refpool(y)

DataAPI.levels(y)

using DataFrames

y = categorical([1,2,3,1])
