# JLBoost.jl

This is a 100%-Julia implementation of regression-tree-boosting based heavily on the algorithms published in the XGBoost, LightGBM and Catboost papers.

This is an early WIP.

## Objectives
* A full-featured tree boosting library with fitting and inference support
* Play nice with the Julia ecosystem e.g. DataFrames.jl and CategoricalArrays.jl
* 100%-Julia
* Fit models on large data
* "Easy" to deploy

## Example
This is very WIP so the API is not stable yet.

```julia
using JLBoost, RDatasets
iris = dataset("datasets", "iris")

iris[!, :is_setosa] = iris[!, :Species] .== "setosa"
target = :is_setosa
features = setdiff(names(iris), [:Species, :is_setosa, :prev_w])
iris[!, :prev_w] .= 0.0

xgtree = jlboost!(iris, target, features; nrounds = 2, maxdepth = 2)

iris.pred = predict(xgtree, iris)
```

## Notes

Currently has a CPU implementation of the `xgboost` binary boosting algorithm as described in the original paper. I am trying to implement the algorithms in the original `xgboost` paper. I want to implement the algorithms mentioned in LigthGBM and Catboost and to port them to GPUs.

There is a similar project called [JuML.jl](https://github.com/Statfactory/JuML.jl).
