# JLBoost.jl

This is a 100%-Julia implementation of regression-tree-gradient-boosting algorithm based heavily on the algorithms published in the XGBoost, LightGBM and Catboost papers.

## Limitations for now
* Currently, only the single-valued models are supported. Multivariate-target models support are *planned*.
* Currently, only the numeric and boolean features are supported. Categorical support are *planned*.

## Objectives
* A full-featured tree boosting library with fitting and inference support
* Play nice with the Julia ecosystem e.g. DataFrames.jl and CategoricalArrays.jl
* 100%-Julia
* Fit models on large data
* Easy to manipulate the tree after fitting; play with tree pruning and adjustments
* "Easy" to deploy

## Example

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
typeof(trees(xgtreemodel)) # Array{JLBoostTree,1}
typeof(xgtreemodel.loss)
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
     -- SepalLength <= 5.1
       ---- weight = -1.1353352832366141

     -- SepalLength > 5.1
       ---- weight = -1.1353352832366104
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





There are also convenience functions for computing the AUC.
````julia
AUC(-iris.pred1, iris.is_setosa)[1]
````


````
0.6666666666666667
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
1×4 DataFrame
│ Row │ feature     │ Quality_Gain │ Coverage │ Frequency │
│     │ Symbol      │ Float64      │ Float64  │ Float64   │
├─────┼─────────────┼──────────────┼──────────┼───────────┤
│ 1   │ PetalLength │ 1.0          │ 1.0      │ 1.0       │
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
jlboost(df, target, features, warm_start, loss)
````


````
JLBoostTreeModel(JLBoostTree[
   -- x <= 52.260169545481986
     -- x <= 29.97479018917961
       -- x <= 10.912271225992388
         -- x <= 6.093034964369215
           -- x <= 3.1692688685016845
             -- x <= 0.20618603306170602
               ---- weight = 0.4921032444134241

             -- x > 0.20618603306170602
               ---- weight = 6.60688826097512

           -- x > 3.1692688685016845
             -- x <= 5.319608657336006
               ---- weight = 10.854554528793017

             -- x > 5.319608657336006
               ---- weight = 12.848396186981947

         -- x > 6.093034964369215
           -- x <= 9.813421897466167
             -- x <= 9.32968358179027
               ---- weight = 18.77711614225921

             -- x > 9.32968358179027
               ---- weight = 20.268085244330276

           -- x > 9.813421897466167
             ---- weight = 21.981228756164

       -- x > 10.912271225992388
         -- x <= 23.091453006070385
           -- x <= 18.917195218406825
             -- x <= 15.733169429191474
               ---- weight = 31.36425359340216

             -- x > 15.733169429191474
               ---- weight = 37.866079651467594

           -- x > 18.917195218406825
             -- x <= 21.74986709763289
               ---- weight = 42.860539418124006

             -- x > 21.74986709763289
               ---- weight = 46.04792835242537

         -- x > 23.091453006070385
           -- x <= 27.692718467555586
             -- x <= 24.15803501692162
               ---- weight = 48.72565865103245

             -- x > 24.15803501692162
               ---- weight = 54.37118615793802

           -- x > 27.692718467555586
             -- x <= 28.395204749646872
               ---- weight = 57.49239957493736

             -- x > 28.395204749646872
               ---- weight = 60.437793562454445

     -- x > 29.97479018917961
       -- x <= 39.99176675167373
         -- x <= 35.29384627470606
           -- x <= 30.957937127922676
             ---- weight = 62.66568297454578

           -- x > 30.957937127922676
             -- x <= 34.14784304236109
               ---- weight = 68.80461542250352

             -- x > 34.14784304236109
               ---- weight = 71.53228801102044

         -- x > 35.29384627470606
           -- x <= 37.69462481425207
             ---- weight = 75.57285757935107

           -- x > 37.69462481425207
             -- x <= 39.73622695593628
               ---- weight = 79.53899673707889

             -- x > 39.73622695593628
               ---- weight = 80.3177590060489

       -- x > 39.99176675167373
         -- x <= 46.169550071865096
           -- x <= 43.12469722077954
             -- x <= 42.653511786716926
               ---- weight = 85.2896278078864

             -- x > 42.653511786716926
               ---- weight = 86.52358871814249

           -- x > 43.12469722077954
             -- x <= 44.76898484587342
               ---- weight = 89.56134807814905

             -- x > 44.76898484587342
               ---- weight = 91.89785771191453

         -- x > 46.169550071865096
           -- x <= 49.173565010228245
             -- x <= 47.89540986442324
               ---- weight = 96.24634077592916

             -- x > 47.89540986442324
               ---- weight = 98.13446754972347

           -- x > 49.173565010228245
             -- x <= 50.14876780521868
               ---- weight = 100.61320557646755

             -- x > 50.14876780521868
               ---- weight = 103.49131493772332

   -- x > 52.260169545481986
     -- x <= 81.66269360031897
       -- x <= 71.21154797707916
         -- x <= 57.30111965932056
           -- x <= 57.14335680080571
             ---- weight = 114.87358810705476

           -- x > 57.14335680080571
             ---- weight = 115.4757434459012

         -- x > 57.30111965932056
           -- x <= 68.45609113875315
             -- x <= 64.76936042514627
               ---- weight = 129.76984076573316

             -- x > 64.76936042514627
               ---- weight = 135.81672093042062

           -- x > 68.45609113875315
             -- x <= 70.02612378686798
               ---- weight = 140.59406796272626

             -- x > 70.02612378686798
               ---- weight = 142.51121832590286

       -- x > 71.21154797707916
         -- x <= 77.51086130774578
           -- x <= 74.89985139662889
             -- x <= 73.83836581427445
               ---- weight = 148.10082856663934

             -- x > 73.83836581427445
               ---- weight = 149.9068967558079

           -- x > 74.89985139662889
             -- x <= 76.35975152375403
               ---- weight = 152.8528137061105

             -- x > 76.35975152375403
               ---- weight = 155.48753117679837

         -- x > 77.51086130774578
           -- x <= 79.81210924353597
             -- x <= 79.17238502577244
               ---- weight = 158.67044284113175

             -- x > 79.17238502577244
               ---- weight = 160.3003305184613

           -- x > 79.81210924353597
             -- x <= 80.9133210613117
               ---- weight = 162.0208124488716

             -- x > 80.9133210613117
               ---- weight = 163.51656396089038

     -- x > 81.66269360031897
       -- x <= 92.14977278002574
         -- x <= 84.45758232903357
           ---- weight = 169.12107702876793

         -- x > 84.45758232903357
           -- x <= 89.7316766242989
             -- x <= 89.12149647608905
               ---- weight = 179.10674351578976

             -- x > 89.12149647608905
               ---- weight = 180.40731678248815

           -- x > 89.7316766242989
             -- x <= 91.1080550177771
               ---- weight = 182.65535449941623

             -- x > 91.1080550177771
               ---- weight = 184.16411631108514

       -- x > 92.14977278002574
         -- x <= 95.1212535414281
           -- x <= 94.97795314494262
             ---- weight = 190.34147048278257

           -- x > 94.97795314494262
             ---- weight = 191.01957346395434

         -- x > 95.1212535414281
           -- x <= 98.921836267936
             ---- weight = 198.39977673110297

           -- x > 98.921836267936
             -- x <= 99.38685517034554
               ---- weight = 199.1142185475966

             -- x > 99.38685517034554
               ---- weight = 199.82472962006074
], LPDistLoss{2}(), :y)
````






### Fit model on `JDF.JDFFile`
Sometimes, you may want to fit a model on a dataset that is too large to fit into RAM. You can convert the dataset to [JDF format](https://github.com/xiaodaigh/JDF.jl) and then use `JDF.JDFFile` functionalities to fit the models. The interface `jlbosst` for `DataFrame` and `JDFFiLe` are the same.

The key advantage of fitting a model using `JDF.JDFFile` is that not all the data need to be loaded into memory and hence will enable large models to be trained on a single computer.

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
AUC(-predict(xgtree1, irisdisk), irisdisk[!, :is_setosa])[1]

# clean up
rm("iris.jdf", force=true, recursive=true)
````






## Notes

Currently has a CPU implementation of the `xgboost` binary boosting algorithm as described in the original paper. I am trying to implement the algorithms in the original `xgboost` paper. I want to implement the algorithms mentioned in LigthGBM and Catboost and to port them to GPUs.

There is a similar project called [JuML.jl](https://github.com/Statfactory/JuML.jl).
