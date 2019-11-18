using JLBoost, RDatasets, JDF
using JDF

using CSV
@time a = CSV.read("c:/data/feature_matrix_cleaned.csv")

savejdf(a, "c:/data/feature_matrix_cleaned.jdf")
a = loadjdf("c:/data/feature_matrix_cleaned.jdf")
type_compress!(a, compress_float=true)
savejdf(a, "c:/data/feature_matrix_cleaned.jdf")

import JDF:some_elm


create_missing!(df, col::Symbol) = begin
	df[!, Symbol(string(col)*"_missing")] = ismissing.(df[!, col])
	if eltype(df[!, col]) <: Union{String, Missing}
		df[!, col] = disallowmissing(coalesce.(df[!, col], "JULIA.MISSING"))
	else
		df[!, col] = disallowmissing(coalesce.(df[!, col], zero(eltype(df[!, col]))))
	end
	df
end

@time a = loadjdf("c:/data/feature_matrix_cleaned.jdf")

using StatsBase
t = [eltype(a) for a in eachcol(a)]
countmap(t)
tt = [Missing <: t for t in t]

using Missings
@time for n in names(a)[tt]
	create_missing!(a, n)
end

droppable = [length(Set(a)) for in eachcol(a)]

savejdf("c:/data/feature_matrix_cleaned_adj.jdf", a)

using JDF, JLBoost
aa = JDFFile("c:/data/feature_matrix_cleaned_adj.jdf")
features = setdiff(names(aa), (:target, :target_missing,  :percentile_target_, :percentile_target__missing, :sk_id_curr, :rowid))
@time xg = jlboost(aa, :target, features; subsample = 0.1, nrounds = 1, max_depth=2, verbose=true)
@time xg1 = jlboost(aa, :target, features, xg; subsample = 0.1, nrounds = 1, max_depth=2, verbose=true)

using Serialization
serialize("c:/data/xg.tree", xg)
serialize("c:/data/xg1.tree", xg1)

deserialize("c:/data/xg.tree")
deserialize("c:/data/xg1.tree")

xg2 = vcat(xg, xg1)

@time pred = predict(xg2, aa)

using DataFrames
aaa = hcat(aa[!, :target], DataFrame(pred = pred))

JLBoost.gini(aaa.x1, aaa.pred)
JLBoost.gini(aaa.pred, aaa.x1)

using Statistics
by(aaa, :x1, pred = :pred => mean)
