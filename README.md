# No more development; not even bug fixes in the foreseeable future

Due to life changes. I have 0 time now to handle this Open Source project. So this will be archived
until I can come back to it.

I will refocus my energy on only a couple of open source packages one of them being {disk.frame}.


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
* Completely [hackable](https://docs.google.com/presentation/d/1xjhi8AbOpBzCxoLy9kGR_NuBqo0O2EWpqQFfXCAjCGY/edit?usp=sharing)

## Quick-start

### Fit model on `DataFrame`

#### Binary Classification
We fit the model by predicting one of the iris Species. To fit a model on a `DataFrame` you need to specify the column and the features default to all columns other than the target.

```julia
using JLBoost, RDatasets
iris = dataset("datasets", "iris");

iris[!, :is_setosa] = iris[!, :Species] .== "setosa";
target = :is_setosa;

features = setdiff(names(iris), ["Species", "is_setosa"]);

# fit one tree
# ?jlboost for more details
xgtreemodel = jlboost(iris, target)
```

```
Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 133.33333333333334, lweight = 2.0, rweight = -2.0, further_split = true
))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 5.8, cutpt = 0, gain
 = 0.0, lweight = 2.0, rweight = 2.0, further_split = false),  => (feature 
= :SepalLength, split_at = 7.9, cutpt = 0, gain = 0.0, lweight = -2.0, rwei
ght = -2.0, further_split = false))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 5.8, cutpt = 0, gain
 = 0.0, lweight = 2.0, rweight = 2.0, further_split = false),  => (feature 
= :SepalLength, split_at = 7.9, cutpt = 0, gain = 0.0, lweight = -2.0, rwei
ght = -2.0, further_split = false))
node to split is next line

JLBoostTreeModel(AbstractJLBoostTree[eta = 1.0 (tree weight)

   -- PetalLength <= 1.9
   -- PetalLength > 1.9], LogitLogLoss(), :is_setosa)
```





The returned model contains a vector of trees and the loss function and target

```julia
typeof(trees(xgtreemodel))
```

```
Vector{AbstractJLBoostTree} (alias for Array{AbstractJLBoostTree, 1})
```



```julia
typeof(xgtreemodel.loss)
```

```
LogitLogLoss
```



```julia
typeof(xgtreemodel.target)
```

```
Symbol
```





You can control parameters like  `max_depth` and `nrounds`
```julia
xgtreemodel2 = jlboost(iris, target; nrounds = 2, max_depth = 2)
```

```
Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 133.33333333333334, lweight = 2.0, rweight = -2.0, further_split = true
))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 5.8, cutpt = 0, gain
 = 0.0, lweight = 2.0, rweight = 2.0, further_split = false),  => (feature 
= :SepalLength, split_at = 7.9, cutpt = 0, gain = 0.0, lweight = -2.0, rwei
ght = -2.0, further_split = false))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 5.8, cutpt = 0, gain
 = 0.0, lweight = 2.0, rweight = 2.0, further_split = false),  => (feature 
= :SepalLength, split_at = 7.9, cutpt = 0, gain = 0.0, lweight = -2.0, rwei
ght = -2.0, further_split = false))
node to split is next line

Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 18.04470443154842, lweight = 1.1353352832366148, rweight = -1.135335283
236615, further_split = true))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 7.9, cutpt = 0, gain
 = 1.7763568394002505e-15, lweight = -1.1353352832366106, rweight = -1.1353
352832366106, further_split = false),  => (feature = :SepalLength, split_at
 = 4.8, cutpt = 16, gain = 8.881784197001252e-16, lweight = 1.1353352832366
135, rweight = 1.1353352832366155, further_split = true))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 7.9, cutpt = 0, gain
 = 1.7763568394002505e-15, lweight = -1.1353352832366106, rweight = -1.1353
352832366106, further_split = false), 
   -- SepalLength <= 4.8
     ---- weight = 1.1353352832366135

   -- SepalLength > 4.8
     ---- weight = 1.1353352832366155
 => (feature = :SepalLength, split_at = 4.8, cutpt = 16, gain = 8.881784197
001252e-16, lweight = 1.1353352832366135, rweight = 1.1353352832366155, fur
ther_split = true))
node to split is next line

JLBoostTreeModel(AbstractJLBoostTree[eta = 1.0 (tree weight)

   -- PetalLength <= 1.9
   -- PetalLength > 1.9, eta = 1.0 (tree weight)

   -- PetalLength <= 1.9
     -- SepalLength <= 4.8
       ---- weight = 1.1353352832366135

     -- SepalLength > 4.8
       ---- weight = 1.1353352832366155

   -- PetalLength > 1.9], LogitLogLoss(), :is_setosa)
```





To grow the tree a leaf-wise (AKA best-first or or in XGBoost terminology "lossguided") strategy,
you see set the `max_leaves` parameters e.g.
```julia
xgtreemodel3 = jlboost(iris, target; nrounds = 2, max_leaves = 8, max_depth = 0)
```

```
Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 133.33333333333334, lweight = 2.0, rweight = -2.0, further_split = true
))
node to split is next line

Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 18.04470443154842, lweight = 1.1353352832366148, rweight = -1.135335283
236615, further_split = true))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 7.9, cutpt = 0, gain
 = 1.7763568394002505e-15, lweight = -1.1353352832366106, rweight = -1.1353
352832366106, further_split = false),  => (feature = :SepalLength, split_at
 = 4.8, cutpt = 16, gain = 8.881784197001252e-16, lweight = 1.1353352832366
135, rweight = 1.1353352832366155, further_split = true))
node to split is next line

JLBoostTreeModel(AbstractJLBoostTree[eta = 1.0 (tree weight)

   -- PetalLength <= 1.9
   -- PetalLength > 1.9, eta = 1.0 (tree weight)

   -- PetalLength <= 1.9
   -- PetalLength > 1.9], LogitLogLoss(), :is_setosa)
```




it recommended that you set `max_depth = 0` to avoid a warning message.


Convenience `predict` function is provided. It can be used to score a tree or a vector of trees
```julia
iris.pred1 = JLBoost.predict(xgtreemodel, iris);
iris.pred2 = JLBoost.predict(xgtreemodel2, iris);
iris.pred1_plus_2 = JLBoost.predict(vcat(xgtreemodel, xgtreemodel2), iris)
```

```
150-element Vector{Float64}:
  5.135335283236616
  5.135335283236616
  5.135335283236613
  5.135335283236613
  5.135335283236616
  5.135335283236616
  5.135335283236613
  5.135335283236616
  5.135335283236613
  5.135335283236616
  ⋮
 -5.135335283236615
 -5.135335283236615
 -5.135335283236615
 -5.135335283236615
 -5.135335283236615
 -5.135335283236615
 -5.135335283236615
 -5.135335283236615
 -5.135335283236615
```





There are also convenience functions for computing the AUC and gini
```julia
AUC(-iris.pred1, iris.is_setosa)
```

```
0.6666666666666667
```



```julia
gini(-iris.pred1, iris.is_setosa)
```

```
0.3333333333333335
```





As a convenience feature, you can adjust the `eta` weight of each tree by multiplying it by a factor e.g.

```Julia
new_tree = 0.3 * trees(xgtreemodel)[1] # weight the first tree by 30%
unique(predict(new_tree, iris) ./ predict(trees(xgtreemodel)[1], iris)) # 0.3
```

#### Feature Importances
One can obtain the feature importance using the `feature_importance` function

```julia
feature_importance(xgtreemodel2, iris)
```

```
2×4 DataFrame
 Row │ feature      Quality_Gain  Coverage  Frequency
     │ Symbol       Float64       Float64?  Float64?
─────┼────────────────────────────────────────────────
   1 │ PetalLength           1.0  0.857143   0.666667
   2 │ SepalLength           0.0  0.142857   0.333333
```





#### Tables.jl integration

Any Tables.jl compatible tabular data structure. So you can use any column accessible table with JLBoost. However, you are advised to define the following methods for `df` as the generic implementation in this package may not be efficient

```julia
nrow(df) # returns the number of rows
ncol(df)
view(df, rows, cols)
```



#### Regression Model
By default `JLBoost.jl` defines it's own `LogitLogLoss` type for  binary classification problems. You may replace the `loss` function-type from the `LossFunctions.jl` `SupervisedLoss` type. E.g for regression models you can choose the leaast squares loss called `L2DistLoss()`

```julia
using DataFrames
using JLBoost
df = DataFrame(x = rand(100) * 100);

df[!, :y] = 2*df.x .+ rand(100);

target = :y;
features = [:x];
warm_start = fill(0.0, nrow(df));


using LossFunctions: L2DistLoss;
loss = L2DistLoss();
jlboost(df, target, features, warm_start, loss; max_depth=2) # default max_depth = 6
```

```
Dict{Any, Any}( => (feature = :x, split_at = 48.63911389031873, cutpt = 45,
 gain = 454424.2872906602, lweight = -51.39571079243722, rweight = -147.209
53355374107, further_split = true))
node to split is next line

Dict{Any, Any}( => (feature = :x, split_at = 74.01487196476889, cutpt = 31,
 gain = 70634.58388926741, lweight = -124.91301137485775, rweight = -176.00
920803479855, further_split = true),  => (feature = :x, split_at = 24.96600
5072729512, cutpt = 23, gain = 51671.5644316056, lweight = -27.961411458694
784, rweight = -75.89520555044068, further_split = true))
node to split is next line

Dict{Any, Any}( => (feature = :x, split_at = 74.01487196476889, cutpt = 31,
 gain = 70634.58388926741, lweight = -124.91301137485775, rweight = -176.00
920803479855, further_split = true), 
   -- x <= 24.966005072729512
     ---- weight = -27.961411458694784

   -- x > 24.966005072729512
     ---- weight = -75.89520555044068
 => (feature = :x, split_at = 24.966005072729512, cutpt = 23, gain = 51671.
5644316056, lweight = -27.961411458694784, rweight = -75.89520555044068, fu
rther_split = true))
node to split is next line

JLBoostTreeModel(AbstractJLBoostTree[eta = 1.0 (tree weight)

   -- x <= 48.63911389031873
     -- x <= 24.966005072729512
       ---- weight = -27.961411458694784

     -- x > 24.966005072729512
       ---- weight = -75.89520555044068

   -- x > 48.63911389031873
     -- x <= 74.01487196476889
       ---- weight = -124.91301137485775

     -- x > 74.01487196476889
       ---- weight = -176.00920803479855
], L2DistLoss(), :y)
```





### Save & Load models
You save the models using the `JLBoost.save` and load it with the `load` function

```julia
JLBoost.save(xgtreemodel, "model.jlb");
JLBoost.save(trees(xgtreemodel), "model_tree.jlb");
```


```julia
JLBoost.load("model.jlb");
JLBoost.load("model_tree.jlb");
```




### Fit model on `JDF.JDFFile` - enabling larger-than-RAM model fit
Sometimes, you may want to fit a model on a dataset that is too large to fit into RAM. You can convert the dataset to [JDF format](https://github.com/xiaodaigh/JDF.jl) and then use `JDF.JDFFile` functionalities to fit the models. The interface `jlbosst` for `DataFrame` and `JDFFiLe` are the same.

The key advantage of fitting a model using `JDF.JDFFile` is that not all the data need to be loaded into memory. This is because `JDF` can load the columns one at a time. Hence this will enable larger models to be trained on a single computer.

```julia
using JLBoost, RDatasets, JDF
iris = dataset("datasets", "iris");

iris[!, :is_setosa] = iris[!, :Species] .== "setosa";
target = :is_setosa;

features = setdiff(Symbol.(names(iris)), [:Species, :is_setosa]);

savejdf("iris.jdf", iris);
irisdisk = JDFFile("iris.jdf");

# fit using on disk JDF format
xgtree1 = jlboost(irisdisk, target, features);
xgtree2 = jlboost(iris, target, features; nrounds = 2, max_depth = 2);

# predict using on disk JDF format
iris.pred1 = predict(xgtree1, irisdisk);
iris.pred2 = predict(xgtree2, irisdisk);

# AUC
AUC(-predict(xgtree1, irisdisk), irisdisk[:, :is_setosa]);

# gini
gini(-predict(xgtree1, irisdisk), irisdisk[:, :is_setosa]);

# clean up
rm("iris.jdf", force=true, recursive=true);
```

```
Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 133.33333333333334, lweight = 2.0, rweight = -2.0, further_split = true
))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 5.8, cutpt = 0, gain
 = 0.0, lweight = 2.0, rweight = 2.0, further_split = false),  => (feature 
= :SepalLength, split_at = 7.9, cutpt = 0, gain = 0.0, lweight = -2.0, rwei
ght = -2.0, further_split = false))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 5.8, cutpt = 0, gain
 = 0.0, lweight = 2.0, rweight = 2.0, further_split = false),  => (feature 
= :SepalLength, split_at = 7.9, cutpt = 0, gain = 0.0, lweight = -2.0, rwei
ght = -2.0, further_split = false))
node to split is next line

Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 133.33333333333334, lweight = 2.0, rweight = -2.0, further_split = true
))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 5.8, cutpt = 0, gain
 = 0.0, lweight = 2.0, rweight = 2.0, further_split = false),  => (feature 
= :SepalLength, split_at = 7.9, cutpt = 0, gain = 0.0, lweight = -2.0, rwei
ght = -2.0, further_split = false))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 5.8, cutpt = 0, gain
 = 0.0, lweight = 2.0, rweight = 2.0, further_split = false),  => (feature 
= :SepalLength, split_at = 7.9, cutpt = 0, gain = 0.0, lweight = -2.0, rwei
ght = -2.0, further_split = false))
node to split is next line

Dict{Any, Any}( => (feature = :PetalLength, split_at = 1.9, cutpt = 50, gai
n = 18.04470443154842, lweight = 1.1353352832366148, rweight = -1.135335283
236615, further_split = true))
node to split is next line

Dict{Any, Any}( => (feature = :SepalLength, split_at = 4.8, cutpt = 16, gai
n = 8.881784197001252e-16, lweight = 1.1353352832366135, rweight = 1.135335
2832366155, further_split = true),  => (feature = :SepalLength, split_at = 
7.9, cutpt = 0, gain = 1.7763568394002505e-15, lweight = -1.135335283236610
6, rweight = -1.1353352832366106, further_split = false))
node to split is next line

Dict{Any, Any}(
   -- SepalLength <= 4.8
     ---- weight = 1.1353352832366135

   -- SepalLength > 4.8
     ---- weight = 1.1353352832366155
 => (feature = :SepalLength, split_at = 4.8, cutpt = 16, gain = 8.881784197
001252e-16, lweight = 1.1353352832366135, rweight = 1.1353352832366155, fur
ther_split = true),  => (feature = :SepalLength, split_at = 7.9, cutpt = 0,
 gain = 1.7763568394002505e-15, lweight = -1.1353352832366106, rweight = -1
.1353352832366106, further_split = false))
node to split is next line
```





### MLJ.jl

Integration with MLJ.jl is available via the [JLBoostMLJ.jl](https://github.com/xiaodaigh/JLBoostMLJ.jl) package

### Hackable

## Notes

Currently has a CPU implementation of the `xgboost` binary boosting algorithm as described in the original paper. I am trying to implement the algorithms in the original `xgboost` paper. I want to implement the algorithms mentioned in LigthGBM and Catboost and to port them to GPUs.

There are two similar projects

* [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl)
* [JuML.jl](https://github.com/Statfactory/JuML.jl)
