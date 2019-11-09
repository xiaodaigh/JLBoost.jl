using JLBoost
using DataFrames#, StatsBase#, ForwardDiff
#using RCall
using JDF
# using Plots
# gr()

# set up global parameters
lambda = 1
gamma  = 3

if !isdir("c:/data/full/gmsc/cs-training.jdf")
    using CSV
    @time df = CSV.read("c:/data/full/gmsc/cs-training.csv")
    rename!(df,Symbol("NumberOfTime30-59DaysPastDueNotWorse") => :NumberOfTime30_59DaysPastDueNotWorse,Symbol("NumberOfTime60-89DaysPastDueNotWorse") => :NumberOfTime60_89DaysPastDueNotWorse);
    savejdf(df, "c:/data/full/gmsc/cs-training.jdf")
end

@time df = loadjdf("c:/data/full/gmsc/cs-training.jdf")

na2n(x) = x == "NA" ? Int32(0) : parse(Int32, x)
na2n(x::Integer) = x
df[!, :NumberOfDependents] = na2n.(df[!, :NumberOfDependents]);
df[!, :MonthlyIncome] = na2n.(df[!, :MonthlyIncome]);

[typeof(c[2]) for c in eachcol(df)]

# setup the initial weights
df[!, :prev_w] .= 0.0;

# set a feature to split on
target = :SeriousDlqin2yrs
prev_w = :prev_w

#features = names(df)[3:end-1][1:2]#[[1:4;6:9]]
features = setdiff(names(df), (:prev_w, target))

booster1 = xgboost(df, target, features; lambda = lambda, gamma = gamma, maxdepth = 3)

loss_fn = JLBoost.logloss
df = df
feature=features[1]
target=target
prev_w=prev_w
lambda
gamma

import JLBoost:g, h

################################################################################
println(feature)
df2 = sort(df[!, [target, feature, prev_w]], feature)

x = df2[!, feature];
target_vec = df2[!, target];
prev_w_vec = df2[!, prev_w];

cg = cumsum(g.(loss_fn, target_vec, prev_w_vec))
ch = cumsum(h.(loss_fn, target_vec, prev_w_vec))

max_cg = cg[end]
max_ch = ch[end]

left_split = (cg).^(2) ./(ch .+ lambda)
right_split = (max_cg.-cg).^(2) ./ ((max_ch .- ch) .+ lambda)
no_split = max_cg^2 /(max_ch + lambda)
lrn = left_split .+  right_split .- no_split .- gamma

df2[!,:lrn] = lrn
df2[!,:rn] = 1:size(df)[1]


df_summ = df2[by(df2, feature, rows_to_keep = :rn => maximum).rows_to_keep, :]

maxloc = findmax(df_summ[!,:lrn])

# (x[maxloc[2]], maxloc)
# df2[!,:ok] = x .<= df_summ[maxloc[2],feature]
# by(df2, :ok, df1 -> (sum(df1[target]), size(df1)[1]))

# store the best split for this val
cutpt = df_summ[maxloc[2],:rn]
lweight = -cg[cutpt]/(ch[cutpt]+lambda)
rweight = -(max_cg - cg[cutpt])/(max_ch - ch[cutpt] + lambda)

(feature = feature, best_split = df_summ[maxloc[2],feature], gain = maxloc[1], lweight=lweight, rweight=rweight)
#################################################################################

delete!(df,:new_weight)
df[:new_weight] = 0.0
@time scoretree(df, booster1, :new_weight);
@time (gini2, cudata) = gini(-df[:new_weight], df[target]; plotauc = true);
gini2

df1 = by(df, :new_weight, df1->DataFrame(d=sum(df1[target]), n=size(df1)[1]))
df1[:dr] = df1[:d]./df1[:n];
df1[:pd] = softmax.(df1[:new_weight])
sort(df1,:new_weight)

sort(by(df,:new_weight, df1->(minimum(df1[:age]),maximum(df1[:age]))), :new_weight)

plot(vcat(0,cudata[1]),vcat(0,cudata[2]))
plot!([0,1],[0,1])

@time booster2 = xgboost(df, target, features; prev_w = :new_weight, lambda = lambda, gamma = gamma)
@time scoretree(df, booster2, :new_weight2);
(gini2, cudata) = gini(-(df[:new_weight2].+df[:new_weight]), df[target]; plotauc = true);
gini2
plot(vcat(0,cudata[1]),vcat(0,cudata[2]))
plot!([0,1],[0,1])
df[:new_weight_all] = df[:new_weight2] .+ df[:new_weight]

@time booster3 = xgboost(df, target, features; prev_w = :new_weight_all, lambda = 1, gamma = gamma)
@time scoretree(df, booster3, :new_weight3);
(gini3, cudata) = gini(-(df[:new_weight3].+df[:new_weight_all]), df[target]; plotauc = true);
gini3
plot(vcat(0,cudata[1]),vcat(0,cudata[2]))
plot!([0,1],[0,1])

@time booster4 = xgboost(df, target, features; prev_w = :new_weight_all, lambda = 1, gamma = gamma)
@time scoretree(df, booster4, :new_weight4);
(gini4, cudata) = gini(-(df[:new_weight4].+df[:new_weight_all]), df[target]; plotauc = true);
print(gini4)
plot(vcat(0,cudata[1]),vcat(0,cudata[2]))
plot!([0,1],[0,1])

df[:new_weight_all] .= df[:new_weight_all] .+ df[:new_weight4]
@time booster5 = xgboost(df, target, features; prev_w = :new_weight_all, lambda = 1, gamma = gamma)
@time scoretree(df, booster5, :new_weight5);
(gini4, cudata) = gini(-(df[:new_weight5].+df[:new_weight_all]), df[target]; plotauc = true);
print(gini4)
plot(vcat(0,cudata[1]),vcat(0,cudata[2]))
plot!([0,1],[0,1])

plot([gini2, gini3, gini4])

using DataFrames, RCall, ForwardDiff, StatsBase, CSV
using Plots
gr()

include(joinpath(pwd(),"JLBoostTree.jl"))
include(joinpath(pwd(),"jlboost.jl"))

R"""
iris1 = iris
"""
@rget iris1
iris1;
unique(iris1[:Species])

df = iris1;
df[:issetosa] = t2one.(df[:Species] .== "virginica");

# set up global parameters
lambda = 1
gamma  = 3

@time df = CSV.read(joinpath(pwd(),"data","cs-training.csv"))
rename!(df,
    Symbol("NumberOfTime30-59DaysPastDueNotWorse") => :NumberOfTime30_59DaysPastDueNotWorse,
    Symbol("NumberOfTime60-89DaysPastDueNotWorse") => :NumberOfTime60_89DaysPastDueNotWorse
);

#countmap(df[:SeriousDlqin2yrs])
#([countmap(ismissing.(c[2])) for c in eachcol(df)], [typeof(c[2]) for c in eachcol(df)])

#countmap(df[:NumberOfDependents])
na2n(x) = x == "NA" ? 0 : parse(Int8, x)
df[:NumberOfDependents] = na2n.(df[:NumberOfDependents]);

# setup the initial weights
df[:prev_w] = 0.0
df;

# set a feature to split on
target = :issetosa
feature = :Petal_Length
prev_w = :prev_w

features = [:Sepal_Length, :Sepal_Width, :Petal_Length, :Petal_Width]

# set a feature to split on
target = :SeriousDlqin2yrs
prev_w = :prev_w

features = names(df)[3:end-1][1:10]#[[1:4;6:9]]
#features = [:RevolvingUtilizationOfUnsecuredLines,:NumberRealEstateLoansOrLines]
#features = [:NumberRealEstateLoansOrLines]

function oklah(df, target, feature1; lambda = lambda, gamma = gamma)
     booster1 = xgboost(df, target, feature1; lambda = lambda, gamma = gamma)
     scoretree(df, booster1, :new_weight);
     (gini2, cudata) = gini(-df[:new_weight], df[target]; plotauc = true);
     gini2
end

@time [oklah(df, target, [feature1]; lambda = lambda, gamma = gamma) for feature1 in features]

features = [:MonthlyIncome]

@time booster1 = xgboost(df, target, features; lambda = lambda, gamma = gamma)

@time scoretree(df, booster1, :new_weight);
@time (gini2, cudata) = gini(-df[:new_weight], df[target]; plotauc = true);
gini2

plot(vcat(0,cudata[1]),vcat(0,cudata[2]))
plot!([0,1],[0,1])

ok=sort(by(df, :new_weight, df1->(sum(df1[target])/size(df1)[1], size(df1)[1])), :new_weight)
ok
#plot(ok[:,1], ok[:,2])
ok[:pd] = softmax.(ok[:new_weight])
ok

countmap(df[:new_weight])

@time booster2 = xgboost(df, target, features; prev_w = :new_weight, lambda = lambda, gamma = gamma)

@time scoretree(df, booster2, :new_weight2);

(gini2, cudata) = gini(-(df[:new_weight2].+df[:new_weight]), df[target]; plotauc = true);
gini2

plot(vcat(0,cudata[1]),vcat(0,cudata[2]))
plot!([0,1],[0,1])

df[:new_weight_all] = df[:new_weight2] .+ df[:new_weight]
countmap(df[:new_weight_all])

@time booster3 = xgboost(df, target, features; prev_w = :new_weight_all, lambda = 1, gamma = gamma)
@time scoretree(df, booster3, :new_weight3);

(gini3, cudata) = gini(-(df[:new_weight3].+df[:new_weight_all]), df[target]; plotauc = true);
gini3

plot(vcat(0,cudata[1]),vcat(0,cudata[2]))
plot!([0,1],[0,1])

df[:new_weight_all] .= df[:new_weight_all] .+ df[:new_weight3]
countmap(df[:new_weight_all])

@time booster4 = xgboost(df, target, features; prev_w = :new_weight_all, lambda = 1, gamma = gamma)
@time scoretree(df, booster4, :new_weight4);
(gini4, cudata) = gini(-(df[:new_weight4].+df[:new_weight_all]), df[target]; plotauc = true);
print(gini4)
plot(vcat(0,cudata[1]),vcat(0,cudata[2]))
plot!([0,1],[0,1])

df[:new_weight_all] .= df[:new_weight_all] .+ df[:new_weight4]
countmap(df[:new_weight_all])

@time booster5 = xgboost(df, target, features; prev_w = :new_weight_all, lambda = 1, gamma = gamma)
@time scoretree(df, booster5, :new_weight5);
(gini4, cudata) = gini(-(df[:new_weight5].+df[:new_weight_all]), df[target]; plotauc = true);
print(gini4)
plot(vcat(0,cudata[1]),vcat(0,cudata[2]))
plot!([0,1],[0,1])
