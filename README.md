# JLBoost.jl

This is a 100%-Julia implementation of Gradient Boosting Regresssion Trees (GBRT) based heavily on the algorithms published in the XGBoost, LightGBM and Catboost papers. GBRT is also referred to as Gradient Boosting Decision Tree (GBDT).

## Limitations for now
* Currently, `Union{T, Missing}` feature type is not supported, but is *planned*.
* Currently, only the single-valued models are supported. Multivariate-target models support is *planned*.
* Currently, only the numeric and boolean features are supported. Categorical support is *planned*.
* Currently, weights cannot be provided for each of the records. Support is *planned*.

## Objectives
* A full-featured & batteries included Gradient Boosting Regression Tree library
* Play nice with the Julia ecosystem e.g. Tables.jl, DataFrames.jl and CategoricalArrays.jl
* 100%-Julia
* Fit models on large data

* Easy to manipulate the tree after fitting; play with tree pruning and adjustments
* "Easy" to deploy

## Quick-start

### Fit model on `DataFrame`

#### Binary Classification
We fit the model by predicting one of the iris Species. To fit a model on a `DataFrame` you need to specify the column and the features default to all columns other than the target.

````julia
using JLBoost, RDatasets
iris = dataset("datasets", "iris")

iris[!, :is_setosa] = iris[!, :Species] .== "setosa"
target = :is_setosa

features = setdiff(names(iris), ["Species", "is_setosa"])

# fit one tree
# ?jlboost for more details
xgtreemodel = jlboost(iris, target)
````


````
JLBoostTreeModel(AbstractJLBoostTree[eta = 1.0 (tree weight)

   -- PetalLength <= 1.9
   -- PetalLength > 1.9], LogitLogLoss(), :is_setosa)
````





The returned model contains a vector of trees and the loss function and target

````julia
typeof(trees(xgtreemodel))
````


````
Array{AbstractJLBoostTree,1}
````



````julia
typeof(xgtreemodel.loss)
````


````
LogitLogLoss
````



````julia
typeof(xgtreemodel.target)
````


````
Symbol
````





You can control parameters like  `max_depth` and `nrounds`
````julia
xgtreemodel2 = jlboost(iris, target; nrounds = 2, max_depth = 2)
````


````
Error: MethodError: no method matching _find_best_split(::LogitLogLoss, ::S
ubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}, ::SubArray
{Bool,1,BitArray{1},Tuple{Array{Int64,1}},false}, ::SubArray{Float64,1,Arra
y{Float64,1},Tuple{Array{Int64,1}},false}, ::Int64, ::Int64; verbose=false,
 max_depth=2)
Closest candidates are:
  _find_best_split(::Any, ::Any, ::Any, ::Any, ::Number, ::Number; min_chil
d_weight, verbose) at c:\git\JLBoost\src\find_best_split.jl:71 got unsuppor
ted keyword argument "max_depth"
  _find_best_split(::LogitLogLoss, ::AbstractArray{T,1} where T, !Matched::
CategoricalArray{T,1,V,C,U,U1} where U1 where U where C where V where T, ::
AbstractArray{T,1} where T, ::Number, ::Number; kwargs...) at c:\git\JLBoos
t\src\find_best_split.jl:58
  _find_best_split(::LogitLogLoss, ::AbstractArray{T,1} where T, !Matched::
SubArray{A,B,C,D,E}, ::AbstractArray{T,1} where T, ::Number, ::Number; kwar
gs...) where {A, B, C<:CategoricalArray, D, E} at c:\git\JLBoost\src\find_b
est_split.jl:64
````





Convenience `predict` function is provided. It can be used to score a tree or a vector of trees
````julia
iris.pred1 = predict(xgtreemodel, iris)
iris.pred2 = predict(xgtreemodel2, iris)
````


````
Error: UndefVarError: xgtreemodel2 not defined
````



````julia
iris.pred1_plus_2 = predict(vcat(xgtreemodel, xgtreemodel2), iris)
````


````
Error: UndefVarError: xgtreemodel2 not defined
````





There are also convenience functions for computing the AUC and gini
````julia
AUC(-iris.pred1, iris.is_setosa)
````


````
0.6666666666666667
````



````julia
gini(-iris.pred1, iris.is_setosa)
````


````
0.3333333333333335
````





As a convenience feature, you can adjust the `eta` weight of each tree by multiplying it by a factor e.g.

```Julia
new_tree = 0.3 * trees(xgtreemodel)[1] # weight the first tree by 30%
unique(predict(new_tree, iris) ./ predict(trees(xgtreemodel)[1], iris)) # 0.3
```

#### Feature Importances
One can obtain the feature importance using the `feature_importance` function

````julia
feature_importance(xgtreemodel, iris)
````


````
Error: InterruptException:
````





#### Tables.jl integration

Any Tables.jl compatible tabular data structure. So you can use any column accessible table with JLBoost. However, you are advised to define the following methods for `df` as the generic implementation in this package may not be efficient

````julia

nrow(df) # returns the number of rows
ncol(df)
view(df, rows, cols)
````




#### Regression Model
By default `JLBoost.jl` defines it's own `LogitLogLoss` type for  binary classification problems. You may replace the `loss` function-type from the `LossFunctions.jl` `SupervisedLoss` type. E.g for regression models you can choose the leaast squares loss called `L2DistLoss()`

````julia
using DataFrames
using JLBoost
df = DataFrame(x = rand(100) * 100)

df[!, :y] = 2*df.x .+ rand(100)

target = :y
features = [:x]
warm_start = fill(0.0, nrow(df))


using LossFunctions: L2DistLoss
loss = L2DistLoss()
jlboost(df, target, features, warm_start, loss; max_depth=2) # default max_depth = 6
````


````
Error: MethodError: no method matching _find_best_split(::LossFunctions.LPD
istLoss{2}, ::SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},fal
se}, ::SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}, ::
SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}, ::Int64, 
::Int64; verbose=false, max_depth=2)
Closest candidates are:
  _find_best_split(::Any, ::Any, ::Any, ::Any, ::Number, ::Number; min_chil
d_weight, verbose) at c:\git\JLBoost\src\find_best_split.jl:71 got unsuppor
ted keyword argument "max_depth"
  _find_best_split(!Matched::LogitLogLoss, ::AbstractArray{T,1} where T, !M
atched::CategoricalArray{T,1,V,C,U,U1} where U1 where U where C where V whe
re T, ::AbstractArray{T,1} where T, ::Number, ::Number; kwargs...) at c:\gi
t\JLBoost\src\find_best_split.jl:58
  _find_best_split(!Matched::LogitLogLoss, ::AbstractArray{T,1} where T, !M
atched::SubArray{A,B,C,D,E}, ::AbstractArray{T,1} where T, ::Number, ::Numb
er; kwargs...) where {A, B, C<:CategoricalArray, D, E} at c:\git\JLBoost\sr
c\find_best_split.jl:64
````





### Save & Load models
You save the models using the `JLBoost.save` and load it with the `load` function

````julia
JLBoost.save(xgtreemodel, "model.jlb")
````


````
testing save
````



````julia
JLBoost.save(trees(xgtreemodel), "model_tree.jlb")
````


````
testing save
````



````julia
JLBoost.load("model.jlb")
JLBoost.load("model_tree.jlb")
````


````
Tree 1
eta = 1.0 (tree weight)

   -- PetalLength <= 1.9
   -- PetalLength > 1.9
````





### Fit model on `JDF.JDFFile` - enabling larger-than-RAM model fit
Sometimes, you may want to fit a model on a dataset that is too large to fit into RAM. You can convert the dataset to [JDF format](https://github.com/xiaodaigh/JDF.jl) and then use `JDF.JDFFile` functionalities to fit the models. The interface `jlbosst` for `DataFrame` and `JDFFiLe` are the same.

The key advantage of fitting a model using `JDF.JDFFile` is that not all the data need to be loaded into memory. This is because `JDF` can load the columns one at a time. Hence this will enable larger models to be trained on a single computer.

````julia
using JLBoost, RDatasets, JDF
````


````
Error: InitError: could not load library "C:\Users\RTX2080\.julia\artifacts
\1b170a85de9456e766ecaa5caf73c8ef5986c046\bin\libblosc.dll"
The specified module could not be found. 
during initialization of module Blosc_jll
````



````julia
iris = dataset("datasets", "iris")

iris[!, :is_setosa] = iris[!, :Species] .== "setosa"
target = :is_setosa

features = setdiff(names(iris), [:Species, :is_setosa])

savejdf("iris.jdf", iris)
````


````
Error: UndefVarError: savejdf not defined
````



````julia
irisdisk = JDFFile("iris.jdf")
````


````
Error: UndefVarError: JDFFile not defined
````



````julia

# fit using on disk JDF format
xgtree1 = jlboost(irisdisk, target, features)
````


````
Error: UndefVarError: irisdisk not defined
````



````julia
xgtree2 = jlboost(iris, target, features; nrounds = 2, max_depth = 2)
````


````
Error: MethodError: no method matching _find_best_split(::LogitLogLoss, ::S
ubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}, ::SubArray
{Bool,1,BitArray{1},Tuple{Array{Int64,1}},false}, ::SubArray{Float64,1,Arra
y{Float64,1},Tuple{Array{Int64,1}},false}, ::Int64, ::Int64; verbose=false,
 max_depth=2)
Closest candidates are:
  _find_best_split(::Any, ::Any, ::Any, ::Any, ::Number, ::Number; min_chil
d_weight, verbose) at c:\git\JLBoost\src\find_best_split.jl:71 got unsuppor
ted keyword argument "max_depth"
  _find_best_split(::LogitLogLoss, ::AbstractArray{T,1} where T, !Matched::
CategoricalArray{T,1,V,C,U,U1} where U1 where U where C where V where T, ::
AbstractArray{T,1} where T, ::Number, ::Number; kwargs...) at c:\git\JLBoos
t\src\find_best_split.jl:58
  _find_best_split(::LogitLogLoss, ::AbstractArray{T,1} where T, !Matched::
SubArray{A,B,C,D,E}, ::AbstractArray{T,1} where T, ::Number, ::Number; kwar
gs...) where {A, B, C<:CategoricalArray, D, E} at c:\git\JLBoost\src\find_b
est_split.jl:64
````



````julia

# predict using on disk JDF format
iris.pred1 = predict(xgtree1, irisdisk)
````


````
Error: UndefVarError: xgtree1 not defined
````



````julia
iris.pred2 = predict(xgtree2, irisdisk)
````


````
Error: UndefVarError: xgtree2 not defined
````



````julia

# AUC
AUC(-predict(xgtree1, irisdisk), irisdisk[!, :is_setosa])
````


````
Error: UndefVarError: xgtree1 not defined
````



````julia

# gini
gini(-predict(xgtree1, irisdisk), irisdisk[!, :is_setosa])
````


````
Error: UndefVarError: xgtree1 not defined
````



````julia
# clean up
rm("iris.jdf", force=true, recursive=true)
````





#### MLJ.jl

Integration with MLJ.jl is available via the [JLBoostMLJ.jl](https://github.com/xiaodaigh/JLBoostMLJ.jl) package

## Notes

Currently has a CPU implementation of the `xgboost` binary boosting algorithm as described in the original paper. I am trying to implement the algorithms in the original `xgboost` paper. I want to implement the algorithms mentioned in LigthGBM and Catboost and to port them to GPUs.

There are two similar projects

* [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl)
* [JuML.jl](https://github.com/Statfactory/JuML.jl)
