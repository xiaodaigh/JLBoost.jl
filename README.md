# JLBoost.jl

This is a 100%-Julia implementation of regression-tree-boosting based heavily on the algorithms published in the XGBoost, LightGBM and Catboost papers.

This is an early WIP.

## Objectives
* A full-featured tree boosting library with fitting and inference support
* Play nice with the Julia ecosystem e.g. DataFrames.jl and CategoricalArrays.jl
* 100%-Julia
* Fit models on large data
* Easy to manipulate the tree after fitting
* "Easy" to deploy

## Example

### Fit model on `DataFrame`
This is very WIP so the API is not stable yet.

```julia
using JLBoost, RDatasets
iris = dataset("datasets", "iris")

iris[!, :is_setosa] = iris[!, :Species] .== "setosa"
target = :is_setosa

features = setdiff(names(iris), [:Species, :is_setosa])
xgtree1 = jlboost(iris, target)
xgtree2 = jlboost(iris, target; nrounds = 2, max_depth = 2)

iris.pred1 = predict(xgtree1, iris)
iris.pred2 = predict(xgtree2, iris)
```

### Fit model on `JDF.JDFFile` 
The key advantage of fitting a model using `JDF.JDFFile` is that not all the data need to be loaded into memory and hence will enable large models to be trained on a single computer.

```julia
using JLBoost, RDatasets, JDF
iris = dataset("datasets", "iris")

iris[!, :is_setosa] = iris[!, :Species] .== "setosa"
target = :is_setosa

features = setdiff(names(iris), [:Species, :is_setosa])

savejdf("iris.jdf", iris)
irisdisk = JDFFile("iris.jdf")

# fit using on disk JDF format
xgtree1 = jlboost(irisdisk, target, features)
xgtree2 = jlboost(iris, target, features; nrounds = 2, max_depth = 2)

# predict using on disk JDF format
iris.pred1 = predict(xgtree1, irisdisk)
iris.pred2 = predict(xgtree2, irisdisk)

# clean up
rm("iris.jdf", force=true, recursive=true)
```


## Notes

Currently has a CPU implementation of the `xgboost` binary boosting algorithm as described in the original paper. I am trying to implement the algorithms in the original `xgboost` paper. I want to implement the algorithms mentioned in LigthGBM and Catboost and to port them to GPUs.

There is a similar project called [JuML.jl](https://github.com/Statfactory/JuML.jl).
