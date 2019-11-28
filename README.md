# JLBoost.jl

This is a 100%-Julia implementation of Gradient Boosting Regresssion Trees (GBRT) based heavily on the algorithms published in the XGBoost, LightGBM and Catboost papers. GBRT is also referred to as Gradient Boosting Decision Tree (GBDT).

## Limitations for now
* Currently, `Union{T, Missing}` feature type is not supported, but are *planned*.
* Currently, only the single-valued models are supported. Multivariate-target models support are *planned*.
* Currently, only the numeric and boolean features are supported. Categorical support are *planned*.

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

features = setdiff(names(iris), [:Species, :is_setosa])

# fit one tree
# ?jlboost for more details
xgtreemodel = jlboost(iris, target)
````


````
JLBoostTreeModel(JLBoostTree[
   -- PetalLength <= 1.9
     ---- weight = 2.0

   -- PetalLength > 1.9
     ---- weight = -2.0
], LogitLogLoss(), :is_setosa)
````





The returned model contains a vector of trees and the loss function and target

````julia
typeof(trees(xgtreemodel))
````


````
Array{JLBoostTree,1}
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
JLBoostTreeModel(JLBoostTree[
   -- PetalLength <= 1.9
     ---- weight = 2.0

   -- PetalLength > 1.9
     ---- weight = -2.0
, 
   -- PetalLength <= 1.9
     -- SepalLength <= 4.8
       ---- weight = 1.1353352832366135

     -- SepalLength > 4.8
       ---- weight = 1.1353352832366155

   -- PetalLength > 1.9
     -- SepalLength <= 7.9
       ---- weight = -1.1353352832366106

     -- SepalLength > 7.9
       ---- weight = -1.1353352832366106
], LogitLogLoss(), :is_setosa)
````





Convenience `predict` function is provided. It can be used to score a tree or a vector of trees
````julia
iris.pred1 = predict(xgtreemodel, iris)
iris.pred2 = predict(xgtreemodel2, iris)
iris.pred1_plus_2 = predict(vcat(xgtreemodel, xgtreemodel2), iris)
````


````
150-element Array{Float64,1}:
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
 -5.135335283236611
 -5.135335283236611
 -5.135335283236611
 -5.135335283236611
 -5.135335283236611
 -5.135335283236611
 -5.135335283236611
 -5.135335283236611
 -5.135335283236611
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

#### MLJ.jl & Tables.jl integrations

There is integration with the MLJ.jl modelling framework

````julia
using MLJ
X, y = unpack(iris, x->!(x in [:is_setosa, :Species]), ==(:is_setosa))

using MLJBase
model = JLBoostClassifier()
````


````
JLBoostClassifier(loss = LogitLogLoss(),
                  nrounds = 1,
                  subsample = 1,
                  eta = 1,
                  max_depth = 6,
                  min_child_weight = 1,
                  lambda = 0,
                  gamma = 0,
                  colsample_bytree = 1,) @ 5…55
````



````julia
mljmodel = fit(model, 1, X, y)
````


````
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
Choosing a split on pred1
Choosing a split on pred2
Choosing a split on pred1_plus_2
(feature = :PetalLength, split_at = 1.9, cutpt = 50, gain = 133.33333333333
334, lweight = 2.0, rweight = -2.0)
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
Choosing a split on pred1
Choosing a split on pred2
Choosing a split on pred1_plus_2
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
Choosing a split on pred1
Choosing a split on pred2
Choosing a split on pred1_plus_2
(fitresult = JLBoostTreeModel(JLBoostTree[
   -- PetalLength <= 1.9
     ---- weight = 2.0

   -- PetalLength > 1.9
     ---- weight = -2.0
], LogitLogLoss(), :__y__),
 cache = nothing,
 report = (AUC = 0.16666666666666669,
           feature_importance = 1×4 DataFrame
│ Row │ feature     │ Quality_Gain │ Coverage │ Frequency │
│     │ Symbol      │ Float64      │ Float64  │ Float64   │
├─────┼─────────────┼──────────────┼──────────┼───────────┤
│ 1   │ PetalLength │ 1.0          │ 1.0      │ 1.0       │,),)
````



````julia
predict(model, mljmodel.fitresult, X)
````


````
150-element Array{Float64,1}:
  2.0
  2.0
  2.0
  2.0
  2.0
  2.0
  2.0
  2.0
  2.0
  2.0
  ⋮  
 -2.0
 -2.0
 -2.0
 -2.0
 -2.0
 -2.0
 -2.0
 -2.0
 -2.0
````





#### Feature Importances
One can obtain the feature importance using the `feature_importance` function

````julia
feature_importance(xgtreemodel, iris)
````


````
1×4 DataFrame
│ Row │ feature     │ Quality_Gain │ Coverage │ Frequency │
│     │ Symbol      │ Float64      │ Float64  │ Float64   │
├─────┼─────────────┼──────────────┼──────────┼───────────┤
│ 1   │ PetalLength │ 1.0          │ 1.0      │ 1.0       │
````





##### Tables.jl

The MLJ.jl frameworks allows interaction with any Tables.jl compatible tabular data structure. So you can use any column accessible table with JLBoost. However you need to have these additional methods defined for the table `df`

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
JLBoostTreeModel(JLBoostTree[
   -- x <= 46.46175206896959
     -- x <= 18.99955811116849
       ---- weight = 18.274163190698808

     -- x > 18.99955811116849
       ---- weight = 67.92977388683316

   -- x > 46.46175206896959
     -- x <= 75.29684664522276
       ---- weight = 122.8385837836683

     -- x > 75.29684664522276
       ---- weight = 181.30740260079625
], LPDistLoss{2}(), :y)
````





### Save & Load models
You save the models using the `JLBoost.save` and load it with the `load` function

````julia
JLBoost.save(xgtreemodel, "model.jlb")
JLBoost.save(trees(xgtreemodel), "model_tree.jlb")
````



````julia
JLBoost.load("model.jlb")
JLBoost.load("model_tree.jlb")
````


````
Tree 1

   -- PetalLength <= 1.9
     ---- weight = 2.0

   -- PetalLength > 1.9
     ---- weight = -2.0
````





### Fit model on `JDF.JDFFile` - enabling larger-than-RAM model fit
Sometimes, you may want to fit a model on a dataset that is too large to fit into RAM. You can convert the dataset to [JDF format](https://github.com/xiaodaigh/JDF.jl) and then use `JDF.JDFFile` functionalities to fit the models. The interface `jlbosst` for `DataFrame` and `JDFFiLe` are the same.

The key advantage of fitting a model using `JDF.JDFFile` is that not all the data need to be loaded into memory. This is because `JDF` can load the columns one at a time. Hence this will enable larger models to be trained on a single computer.

````julia
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
AUC(-predict(xgtree1, irisdisk), irisdisk[!, :is_setosa])

# gini
gini(-predict(xgtree1, irisdisk), irisdisk[!, :is_setosa])

# clean up
rm("iris.jdf", force=true, recursive=true)
````





## Why create JLBoost.jl when XGBoost.jl works perfectly fine?

Indeed XGBoost.jl is greate and so are the Python and R incarnations of XGBoost. But something is amiss. I feel like I have to jump through hoops to get it to work. E.g. I have convert the data to matrix format, so I can't use dataframes directly, and it can't feal with CategoricalArrays directly and instead have to employ one hot encoding. LightGBM has interesting algorithms for deal with categorical features but they are not so easy to implement in XGBoost.jl as that's basicaly a C++ library.

The journey that lead to create JLBoost.jl was an interesting one. When I was running the Sydney Julia meetup (SJM), I got an email from Adam, the creator of JuML.jl, and we discussed the possibility of giving a presentation at SJM. Adam mentioned that he created an XGBoost clone in Julia that was faster than the C++ XGBoost implementation. Hist implemnetaion was only about 500-600 lines of Julia and has a much smaller memory footprint. I was slightly skeptical as you can imagine. When I first came across XGBoost, I knew it as a C++ library. And because it had help win so many Kaggle competitions, I thought it must contain some highly advanced math that I won't really understand even though I have an honours degree in pure maths and a master degree in statistics. In other words, I was intimated by XGBoost.

But Adam made me curious. He mentioned that I should just read the XGBoost paper. So I did. Initially, I struggle, but soon everything fell into place. I can actually understand this! In fact, I can understand this well enough that I can explain it to others. And it isn't because I am super smart, it's because The boosting algorithm described in the XGBoost paper rquires some high school level algebra and an understanding of basic calculus and Taylor series expansions, which are typically covered in first or second year of university.

JuML.jl opened my eyes to the possibilities of Julia, but it only implemented the binary logistic case and it didn't work with DataFrames.jl. Beyond JuML.jl, But there was no pure-Julia implementation of the XGBoost algorithms. So I have this idea to implement JLBoost that layed dormant as I slogged away at my day time job as a consultant.

Recently, I decided to finally give JLBoost a crack! And I was able to implement the basic XGBoost algorithms, use DataFrames.jl, allow on-disk fitting with JDF.Files, all in a few hundred lines of code!

Doing JLBoost.jl in Julia makes the package more extensible. In fact, I can implement the Tabless.jl interface and allow any Tables.jl-compatible data structure to fit models using JLBoost! I can add support to any scalar target models easily without having to resort to C++. In fact, any sufficiently motivated Julia-programming can enjoy JLBoost.jl's boosting algorithm if they just implement g and h for their loss function! Neat!

In conclusion, JLBoost.jl being pure-Julia makes the code base much smaller and easier to maintiain. It also makes it much more extensible then equivalent C++ implementations. It can work with Tables.jl data structure that JLBoost.jl doesn't know about.


## Notes

Currently has a CPU implementation of the `xgboost` binary boosting algorithm as described in the original paper. I am trying to implement the algorithms in the original `xgboost` paper. I want to implement the algorithms mentioned in LigthGBM and Catboost and to port them to GPUs.

There is a similar project called [JuML.jl](https://github.com/Statfactory/JuML.jl).
