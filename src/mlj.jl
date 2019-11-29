export fit, predict, fitted_params, JLBoostMLJModel, JLBoostClassifier, JLBoostRegressor, JLBoostCount

#using MLJBase
import MLJBase: Probabilistic, Deterministic, clean!, fit, predict, fitted_params, load_path, Table
import MLJBase: package_name, package_uuid, package_url, is_pure_julia, package_license
import MLJBase: input_scitype, target_scitype, docstring

using ScientificTypes: Continuous, OrderedFactor, Count, Multiclass, Finite

using LossFunctions: PoissonLoss, L2DistLoss

# supervised determinstinistic model
#abstract type JLBoostMLJModel <: Supervised end

mutable struct JLBoostClassifier <: Probabilistic
    loss
    nrounds
    subsample
    eta
    max_depth
    min_child_weight
    lambda
    gamma
    colsample_bytree
end

mutable struct JLBoostRegressor <: Deterministic
    loss
    nrounds
    subsample
    eta
    max_depth
    min_child_weight
    lambda
    gamma
    colsample_bytree
end
mutable struct JLBoostCount <: Deterministic
    loss
    nrounds
    subsample
    eta
    max_depth
    min_child_weight
    lambda
    gamma
    colsample_bytree
end

"""
    JLBoostClassifier(;
        loss = LogitLogLoss(),
        nrounds = 1,
        subsample = 1,
        eta = 1,
        max_depth = 6,
        min_child_weight = 1,
        lambda = 0,
        gamma = 0,
        colsample_bytree = 1)

Return an MLJ.jl compatible Model. The parameters are the same as `jlboost`. See `?jlboost`
"""
JLBoostClassifier(;
    loss = LogitLogLoss(),
    nrounds = 1,
    subsample = 1,
    eta = 1,
    max_depth = 6,
    min_child_weight = 1,
    lambda = 0,
    gamma = 0,
    colsample_bytree = 1) = JLBoostClassifier(loss, nrounds, subsample, eta, max_depth, min_child_weight, lambda, gamma, colsample_bytree)

"""
    JLBoostRegressor(;
        loss = L2DistLoss(),
        nrounds = 1,
        subsample = 1,
        eta = 1,
        max_depth = 6,
        min_child_weight = 1,
        lambda = 0,
        gamma = 0,
        colsample_bytree = 1)

Return an MLJ.jl compatible Model. The parameters are the same as `jlboost`. See `?jlboost`
"""
JLBoostRegressor(;
    loss = L2DistLoss(),
    nrounds = 1,
    subsample = 1,
    eta = 1,
    max_depth = 6,
    min_child_weight = 1,
    lambda = 0,
    gamma = 0,
    colsample_bytree = 1) = JLBoostRegressor(loss, nrounds, subsample, eta, max_depth, min_child_weight, lambda, gamma, colsample_bytree)

"""
    JLBoostCount(;
        loss = L2DistLoss(),
        nrounds = 1,
        subsample = 1,
        eta = 1,
        max_depth = 6,
        min_child_weight = 1,
        lambda = 0,
        gamma = 0,
        colsample_bytree = 1)

Return an MLJ.jl compatible Model. The parameters are the same as `jlboost`. See `?jlboost`
"""
JLBoostCount(;
    loss = PoissonLoss(),
    nrounds = 1,
    subsample = 1,
    eta = 1,
    max_depth = 6,
    min_child_weight = 1,
    lambda = 0,
    gamma = 0,
    colsample_bytree = 1) = JLBoostCount(loss, nrounds, subsample, eta, max_depth, min_child_weight, lambda, gamma, colsample_bytree)

const JLBoostMLJModel = Union{JLBoostClassifier, JLBoostRegressor, JLBoostCount}

# see https://alan-turing-institute.github.io/MLJ.jl/stable/adding_models_for_general_use/#The-fit-method-1
fit(model::JLBoostMLJModel, verbosity::Int, X, y::Vector) = begin
    fit(model::JLBoostMLJModel, verbosity::Integer, X, DataFrame(__y__ = y))
end

fit(model::JLBoostClassifier, verbosity::Int, X, y) = begin
    if typeof(y) <: AbstractVector
        y = DataFrame(__y__ = y)
    end
    df = hcat(X, y)

    target = names(y)[1]
    features = setdiff(names(X), names(y))
    warm_start = fill(0.0, nrow(X))
    fitresult = jlboost(df, target, features, warm_start , model.loss;
        nrounds = model.nrounds, subsample = model.subsample, eta = model.eta,
        colsample_bytree = model.colsample_bytree, max_depth = model.max_depth,
        min_child_weight = model.min_child_weight, lambda = model.lambda,
        gamma = model.gamma, verbose = verbosity >= 1
     )

     if length(levels(y[:, 1])) == 2
         println("meh")
         res = (
            fitresult = fitresult,
            cache = nothing,
            report = (
                AUC2 = abs(AUC(predict(fitresult, X), y[:, 1]))#,
                # feature_importance2 = feature_importance(fitresult, df)
            )
        )
        println("got here")
        return res
    else
         return (
            fitresult = fitresult,
            cache = nothing,
            report = (
                placeholder = true
                # feature_importance3 = feature_importance(fitresult, df)
            )
        )
    end
end

# see https://alan-turing-institute.github.io/MLJ.jl/stable/adding_models_for_general_use/#The-fitted_params-method-1
fitted_params(model::JLBoostMLJModel, fitresult) = (fitresult = fitresult, trees = trees(fitresult))

#  seehttps://alan-turing-institute.github.io/MLJ.jl/stable/adding_models_for_general_use/#The-predict-method-1
predict(model::JLBoostMLJModel, fitresult, Xnew) = begin
    predict(fitresult, Xnew)
end

# see https://alan-turing-institute.github.io/MLJ.jl/stable/adding_models_for_general_use/#Trait-declarations-1
input_scitype(::Type{<:JLBoostMLJModel}) = Table(Union{Continuous, OrderedFactor, Count})

target_scitype(::Type{<:JLBoostClassifier}) = AbstractVector{<:Finite{2}} #AbstractVector{<:Multiclass{2}}
target_scitype(::Type{<:JLBoostRegressor}) = AbstractVector{<:Continuous}
target_scitype(::Type{<:JLBoostCount}) = AbstractVector{<:Count}

# Misc see https://alan-turing-institute.github.io/MLJ.jl/stable/adding_models_for_general_use/
load_path(::Type{JLBoostClassifier}) = "JLBoost.JLBoostClassifier"
load_path(::Type{<:JLBoostRegressor}) = "JLBoost.JLBoostRegressor"
load_path(::Type{<:JLBoostCount}) = "JLBoost.JLBoostCount"


package_name(::Type{<:JLBoostMLJModel}) = "JLBoost"
package_uuid(::Type{<:JLBoostMLJModel}) = "13d6d4a1-5e7f-472c-9ebc-8123a4fbb95f"
package_url(::Type{<:JLBoostMLJModel}) = "https://github.com/xiaodaigh/JLBoost.jl"
is_pure_julia(::Type{<:JLBoostMLJModel}) = true
package_license(::Type{<:JLBoostMLJModel}) = "MIT"

docstring(::Type{JLBoostClassifier}) =
    "The JLBoost gradient boosting method, for use with "*
    "`Binary=Finite{2}` univariate targets"

docstring(::Type{JLBoostRegressor}) =
    "The JLBoost gradient boosting method, for use with "*
    "`Continuous` univariate targets"

docstring(::Type{JLBoostCount}) =
    "The JLBoost gradient boosting method, for use with "*
    "`Count` univariate targets, using a Poisson loss function"
