You can fit Decisions with [JLBoost.jl](https://github.com/xiaodaigh/JLBoost.jl) which has implemented the gradient-boosting regression method in the originla XGBoost paper. It's an 100%-Julia implementation.

**So why create JLBoost.jl when XGBoost.jl?**

1) Play nice with the Julia ecosystem like DataFrames.jl and (soon) CategoricalArrays.jl
2) Demonstrate how a 100%-Julia implementation is superior in terms of maintainability and extensibility
3) Why not?

The model is at MVP (minimal viable product) stage and not MLP (minimum lovable product) stage. So lots of features are missing. But it's working quite well for simple regression tasks! Please try it out and submit issues!

## The journey to JLBoost.jl
Indeed XGBoost.jl is great and so are the Python and R incarnations of XGBoost. But something is amiss. I feel like I have to jump through hoops to get it to work. E.g. I have convert the data to matrix format, so I can't use dataframes directly, and it can't feal with CategoricalArrays directly and instead have to employ one hot encoding. LightGBM has interesting algorithms for deal with categorical features but they are not so easy to implement in XGBoost.jl as that's basicaly a C++ library.

The journey that lead to create JLBoost.jl was an interesting one. When I was running the Sydney Julia meetup (SJM), I got an email from Adam, the creator of JuML.jl, and we discussed the possibility of giving a presentation at SJM. Adam mentioned that he created an XGBoost clone in Julia that was faster than the C++ XGBoost implementation. Hist implemnetaion was only about 500-600 lines of Julia and has a much smaller memory footprint. I was slightly skeptical as you can imagine. When I first came across XGBoost, I knew it as a C++ library. And because it had help win so many Kaggle competitions, I thought it must contain some highly advanced math that I won't really understand even though I have an honours degree in pure maths and a master degree in statistics. In other words, I was intimated by XGBoost.

But Adam made me curious. He mentioned that I should just read the XGBoost paper. So I did. Initially, I struggle, but soon everything fell into place. I can actually understand this! In fact, I can understand this well enough that I can explain it to others. And it isn't because I am super smart, it's because The boosting algorithm described in the XGBoost paper rquires some high school level algebra and an understanding of basic calculus and Taylor series expansions, which are typically covered in first or second year of university.

JuML.jl opened my eyes to the possibilities of Julia, but it only implemented the binary logistic case and it didn't work with DataFrames.jl. Beyond JuML.jl, But there was no pure-Julia implementation of the XGBoost algorithms. So I have this idea to implement JLBoost that layed dormant as I slogged away at my day time job as a consultant.

Recently, I decided to finally give JLBoost a crack! And I was able to implement the basic XGBoost algorithms, use DataFrames.jl, allow on-disk fitting with JDF.Files, all in a few hundred lines of code!

Doing JLBoost.jl in Julia makes the package more extensible. In fact, I can implement the Tabless.jl interface and allow any Tables.jl-compatible data structure to fit models using JLBoost! I can add support to any scalar target models easily without having to resort to C++. In fact, any sufficiently motivated Julia-programming can enjoy JLBoost.jl's boosting algorithm if they just implement g and h for their loss function! Neat!

In conclusion, JLBoost.jl being pure-Julia makes the code base much smaller and easier to maintiain. It also makes it much more extensible then equivalent C++ implementations. It can work with Tables.jl data structure that JLBoost.jl doesn't know about.


## Quick-start

### Fit model on  `DataFrame`

#### Binary Classification

We fit the model by predicting one of the iris Species. To fit a model on a  `DataFrame`  you need to specify the column and the features default to all columns other than the target.

using JLBoost, RDatasets iris = dataset("datasets", "iris") iris[!, :is_setosa] = iris[!, :Species] .== "setosa" target = :is_setosa features = setdiff(names(iris), [:Species, :is_setosa]) # fit one tree # ?jlboost for more details xgtreemodel = jlboost(iris, target)

```
JLBoostTreeModel(JLBoostTree[
   -- PetalLength <= 1.9
     ---- weight = 2.0

   -- PetalLength > 1.9
     ---- weight = -2.0
], LogitLogLoss(), :is_setosa)
```

The returned model contains a vector of trees and the loss function and target

typeof(trees(xgtreemodel))

```
Array{JLBoostTree,1}
```

typeof(xgtreemodel.loss)

```
LogitLogLoss
```

typeof(xgtreemodel.target)

```
Symbol
```

You can control parameters like  `max_depth`  and  `nrounds`

xgtreemodel2 = jlboost(iris, target; nrounds = 2, max_depth = 2)

```
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
```

Convenience  `predict`  function is provided. It can be used to score a tree or a vector of trees

iris.pred1 = predict(xgtreemodel, iris) iris.pred2 = predict(xgtreemodel2, iris) iris.pred1_plus_2 = predict(vcat(xgtreemodel, xgtreemodel2), iris)

```
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
```

There are also convenience functions for computing the AUC and gini

AUC(-iris.pred1, iris.is_setosa)

```
0.6666666666666667
```

gini(-iris.pred1, iris.is_setosa)

```
0.3333333333333335
```

As a convenience feature, you can adjust the  `eta`  weight of each tree by multiplying it by a factor e.g.

new_tree = 0.3 * trees(xgtreemodel)[1] # weight the first tree by 30% unique(predict(new_tree, iris) ./ predict(trees(xgtreemodel)[1], iris)) # 0.3

#### MLJ.jl integrations

There is integration with the MLJ.jl modelling framework

using MLJ X, y = unpack(iris, x->!(x in [:is_setosa, :Species]), ==(:is_setosa)) using MLJBase model = JLBoostModel()

```
JLBoostModel(loss = LogitLogLoss(),
             nrounds = 1,
             subsample = 1,
             eta = 1,
             max_depth = 6,
             min_child_weight = 1,
             lambda = 0,
             gamma = 0,
             colsample_bytree = 1,) @ 1…08
```

mljmodel = fit(model, 1, X, y)

```
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
```

predict(model, mljmodel.fitresult, X)

```
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
```

#### Feature Importances

One can obtain the feature importance using the  `feature_importance`  function

feature_importance(xgtreemodel, iris)

```
1×4 DataFrame
│ Row │ feature     │ Quality_Gain │ Coverage │ Frequency │
│     │ Symbol      │ Float64      │ Float64  │ Float64   │
├─────┼─────────────┼──────────────┼──────────┼───────────┤
│ 1   │ PetalLength │ 1.0          │ 1.0      │ 1.0       │
```

#### Regression Model

By default  `JLBoost.jl`  defines it's own  `LogitLogLoss`  type for binary classification problems. You may replace the  `loss`  function-type from the  `LossFunctions.jl`   `SupervisedLoss`  type. E.g for regression models you can choose the leaast squares loss called  `L2DistLoss()`

using DataFrames using JLBoost df = DataFrame(x = rand(100) * 100) df[!, :y] = 2*df.x .+ rand(100) target = :y features = [:x] warm_start = fill(0.0, nrow(df)) using LossFunctions: L2DistLoss loss = L2DistLoss() jlboost(df, target, features, warm_start, loss; max_depth=2) # default max_depth = 6

```
JLBoostTreeModel(JLBoostTree[
   -- x <= 47.33448875007904
     -- x <= 23.610929988002827
       ---- weight = 22.038150565683793

     -- x > 23.610929988002827
       ---- weight = 75.33926763626144

   -- x > 47.33448875007904
     -- x <= 74.15995188333997
       ---- weight = 124.18955286168057

     -- x > 74.15995188333997
       ---- weight = 175.81849492929982
], LPDistLoss{2}(), :y)
```

### Save & Load models

You save the models using the  `JLBoost.save`  and load it with the  `load`  function

JLBoost.save(xgtreemodel, "model.jlb") JLBoost.save(trees(xgtreemodel), "model_tree.jlb")

JLBoost.load("model.jlb") JLBoost.load("model_tree.jlb")

```
Tree 1

   -- PetalLength <= 1.9
     ---- weight = 2.0

   -- PetalLength > 1.9
     ---- weight = -2.0
```

### Fit model on  `JDF.JDFFile`  - enabling larger-than-RAM model fit

Sometimes, you may want to fit a model on a dataset that is too large to fit into RAM. You can convert the dataset to [JDF format](https://github.com/xiaodaigh/JDF.jl) and then use  `JDF.JDFFile`  functionalities to fit the models. The interface  `jlbosst`  for  `DataFrame`  and  `JDFFiLe`  are the same.

The key advantage of fitting a model using  `JDF.JDFFile`  is that not all the data need to be loaded into memory and hence will enable large models to be trained on a single computer.

using JLBoost, RDatasets, JDF iris = dataset("datasets", "iris") iris[!, :is_setosa] = iris[!, :Species] .== "setosa" target = :is_setosa features = setdiff(names(iris), [:Species, :is_setosa]) savejdf("iris.jdf", iris) irisdisk = JDFFile("iris.jdf") # fit using on disk JDF format xgtree1 = jlboost(irisdisk, target, features) xgtree2 = jlboost(iris, target, features; nrounds = 2, max_depth = 2) # predict using on disk JDF format iris.pred1 = predict(xgtree1, irisdisk) iris.pred2 = predict(xgtree2, irisdisk) # AUC AUC(-predict(xgtree1, irisdisk), irisdisk[!, :is_setosa]) # gini gini(-predict(xgtree1, irisdisk), irisdisk[!, :is_setosa]) # clean up rm("iris.jdf", force=true, recursive=true)
