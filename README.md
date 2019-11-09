# JLBoost.jl

I am trying to implement the algorithms in the original `xgboost` paper. I want to implement the algorithms mentioned in LigthGBM and Catboost and to port them to GPUs.

This is an early WIP. 

## Example
```julia
using JLBoost, RDatasets
iris = dataset("datasets", "iris")

iris[!, :is_setosa] = iris[!, :Species] .== "setosa"
target = :is_setosa
features = setdiff(names(iris), [:Species, :is_setosa, :prev_w])
iris[!, :prev_w] .= 0.0

xgtree = xgboost(iris, target, features; lambda = 1, gamma = 3, maxdepth = 2)
```

## Plans

Implement modelling fitting and inferencing. 

## Notes

Currently has a CPU implementation of the `xgboost` boosting algorithm as described in the original paper.
