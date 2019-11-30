# MLJJLBoost

````julia
using RDatasets
iris = dataset("datasets", "iris")
iris[!, :is_setosa] .= iris.Species .== "setosa"

using MLJ, MLJBase, MLJJLBoost
X, y = unpack(iris, x->!(x in [:is_setosa, :Species]), ==(:is_setosa))

using MLJJLBoost:JLBoostClassifier
model = JLBoostClassifier()
````


````
JLBoostClassifier(loss = JLBoost.LogitLogLoss(),
                  nrounds = 1,
                  subsample = 1,
                  eta = 1,
                  max_depth = 6,
                  min_child_weight = 1,
                  lambda = 0,
                  gamma = 0,
                  colsample_bytree = 1,) @ 2…77
````





Fit the model
````julia
mljmodel = fit(model, 1, X, y)
````


````
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
(feature = :PetalLength, split_at = 1.9, cutpt = 50, gain = 133.33333333333
334, lweight = 2.0, rweight = -2.0)
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
Choosing a split on SepalLength
Choosing a split on SepalWidth
Choosing a split on PetalLength
Choosing a split on PetalWidth
meh
got here
(fitresult = JLBoost.JLBoostTrees.JLBoostTreeModel(JLBoost.JLBoostTrees.JLB
oostTree[
   -- PetalLength <= 1.9
     ---- weight = 2.0

   -- PetalLength > 1.9
     ---- weight = -2.0
], JLBoost.LogitLogLoss(), :__y__),
 cache = nothing,
 report = 0.16666666666666669,)
````





Predicting using the model

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
feature_importance(mljmodel.fitresult, X, y)
````


````
1×4 DataFrame
│ Row │ feature     │ Quality_Gain │ Coverage │ Frequency │
│     │ Symbol      │ Float64      │ Float64  │ Float64   │
├─────┼─────────────┼──────────────┼──────────┼───────────┤
│ 1   │ PetalLength │ 1.0          │ 1.0      │ 1.0       │
````


