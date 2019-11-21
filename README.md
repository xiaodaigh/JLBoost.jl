# JLBoost.jl

This is a 100%-Julia implementation of regression-tree-gradient-boosting algorithm based heavily on the algorithms published in the XGBoost, LightGBM and Catboost papers.

## Limitations for now
* Currently, only the scalar-target models are supported. Multivariate-target models support are *planned*.
* Currently, only the numeric and boolean features are supported. Categorical feature support are *planned*.

## Objectives
* A full-featured gradient boosting regression tree package with model fit and inference support
* Play nice with the Julia ecosystem e.g. DataFrames.jl and CategoricalArrays.jl
* 100%-Julia
* Fit models on large data
* Easy to manipulate the tree after fitting; play with tree pruning and adjustments
* "Easy" to deploy

## Example

### Fit model on `DataFrame`

#### Binary Classification
We fit the model by predicting one of the iris Species. To fit a model on a `DataFrame` you need to specify the column and the features default to all columns other than the target.

```julia
using JLBoost, RDatasets
iris = dataset("datasets", "iris")

iris[!, :is_setosa] = iris[!, :Species] .== "setosa"
target = :is_setosa

features = setdiff(names(iris), [:Species, :is_setosa])

# fit one tree
# ?jlboost for more details
xgtree1 = jlboost(iris, target)
```

The returned model is a vector of trees

```julia
typeof(xgtree1) # Array{JLBoostTree{Float64},1}
```

You can control parameters like  `max_depth` and `nrounds`
```julia
xgtree2 = jlboost(iris, target; nrounds = 2, max_depth = 2)
```

Convenience `predict` function is provided. It can be used to score a tree or a vector of trees
```julia
iris.pred1 = predict(xgtree1, iris)
iris.pred2 = predict(xgtree2, iris)
iris.pred1_plus_2 = predict(vcat(xgtree1, xgtree2), iris)
```

There are also convenience functions for computing the AUC.
```julia
iris.pred1 = predict(xgtree1, iris)
iris.pred2 = predict(xgtree2, iris)
iris.pred1_plus_2 = predict(vcat(xgtree1, xgtree2), iris)
```

As a convenience feature, you can adjust the `eta` weight of each tree by multiplying it by a factor e.g.

```Julia
new_tree = 0.3 * xgtree1[1] # weight the first tree by 30%
unique(predict(new_tree, iris) ./ predict(xgtree[1], iris)) # 0.3
```

#### Regression Model
By default `JLBoost.jl` defines it's own `LogitLogLoss` type for  binary classification problems. You may replace the `loss` function-type with function-type that sub-types `LossFunctions.jl`'s `SupervisedLoss` type. E.g for regression models you can choose the least-squares loss called `L2DistLoss()`

```julia
using DataFrames
using JLBoost
df = DataFrame(x = rand(100) * 100)

df[!, :y] = 2*df.x .+ rand(100)

target = :y
features = [:x]
warm_start = fill(0.0, nrow(df))


using LossFunctions: L2DistLoss
loss = L2DistLoss()
jlboost(df, target, features, warm_start, loss)
```


### Fit model on `JDF.JDFFile`
Sometimes, you may want to fit a model on a dataset that is too large to fit into RAM. You can convert the dataset to [JDF format](https://github.com/xiaodaigh/JDF.jl) and then use `JDF.JDFFile` functionalities to fit the models. The interface `jlbosst` for `DataFrame` and `JDFFiLe` are the same.

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

# AUC
AUC(-predict(xgtree1, irisdisk), irisdisk[!, :Species])[1]

# clean up
rm("iris.jdf", force=true, recursive=true)
```


## Notes

Currently has a CPU implementation of the `xgboost` binary boosting algorithm as described in the original paper. I am trying to implement the algorithms in the original `xgboost` paper. I want to implement the algorithms mentioned in LigthGBM and Catboost and to port them to GPUs.

There is a similar project called [JuML.jl](https://github.com/Statfactory/JuML.jl).
