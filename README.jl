using Pkg
Pkg.activate("c:/git/JLBoost")
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

nrounds = 1

res_jlt = Vector{JLBoostTree{Float64}}(undef, nrounds)

if colsample_bytree < 1
	features_sample = sample(features, ceil(length(features)*colsample_bytree) |> Int; replace = true)
else
	features_sample = features
end

if subsample == 1
	warm_start = fill(0.0, nrow(df))
	new_jlt = _fit_tree!(loss, df, target, features_sample, warm_start, nothing, JLBoostTree(0.0); kwargs...);
else
	rows = sample(collect(1:nrow(df)), Int(round(nrow(df)*subsample)); replace = false)
	warm_start = fill(0.0, length(rows))
	new_jlt = _fit_tree!(loss, @view(df[rows, :]), target, features_sample, warm_start, nothing, JLBoostTree(0.0); kwargs...);
end
res_jlt[1] = deepcopy(new_jlt)

for nround in 2:nrounds
	if verbose
		println("Fitting tree #$(nround)")
	end
	# assign the previous weight

	if colsample_bytree < 1
		features_sample = sample(features, ceil(length(features)*colsample_bytree) |> Int; replace = true)
	else
		features_sample = features
	end

	if subsample == 1
		warm_start = predict(res_jlt[1:nrounds-1], df)
		new_jlt = _fit_tree!(loss, df, target, features_sample, warm_start, nothing, JLBoostTree(0.0); kwargs...);
	else
		rows = sample(collect(1:nrow(df)), Int(round(nrow(df)*subsample)); replace = false)
		warm_start = predict(res_jlt[1:nrounds-1], @view(df[rows, :]))

		new_jlt = _fit_tree!(loss, @view(df[rows, :]), target, features_sample, warm_start, nothing, JLBoostTree(0.0); kwargs...);
	end
    res_jlt[nround] = deepcopy(new_jlt)
end
res_jlt
