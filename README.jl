
using JLBoost, RDatasets
iris = dataset("datasets", "iris")

iris[!, :is_setosa] = iris[!, :Species] .== "setosa"
target = :is_setosa

features = setdiff(names(iris), [:Species, :is_setosa])

# fit one tree
# ?jlboost for more details
xgtreemodel = jlboost(iris, target)


typeof(trees(xgtreemodel))


typeof(xgtreemodel.loss)


typeof(xgtreemodel.target)


xgtreemodel2 = jlboost(iris, target; nrounds = 2, max_depth = 2)


iris.pred1 = predict(xgtreemodel, iris)
iris.pred2 = predict(xgtreemodel2, iris)
iris.pred1_plus_2 = predict(vcat(xgtreemodel, xgtreemodel2), iris)


AUC(-iris.pred1, iris.is_setosa)


gini(-iris.pred1, iris.is_setosa)


feature_importance(xgtreemodel, iris)


nrow(df) # returns the number of rows
ncol(df)
view(df, rows, cols)


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


JLBoost.save(xgtreemodel, "model.jlb")
JLBoost.save(trees(xgtreemodel), "model_tree.jlb")


JLBoost.load("model.jlb")
JLBoost.load("model_tree.jlb")


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

