using Pkg
Pkg.activate(".")
@time using DataFrames
@time using JDF
@time using JLBoost, LossFunctions

###############################################################################
# testing best_split
###############################################################################
if !isdir("c:/data/Churn_Modelling_fnl_w_profit.jdf")
    using CSV
    a = CSV.read("c:/data/Churn_Modelling_fnl_w_profit.csv")
    savejdf(a, "c:/data/Churn_Modelling_fnl_w_profit.jdf")
end

@time a = loadjdf("c:/data/Churn_Modelling_fnl_w_profit.jdf")

@time treem = jlboost(a, :Exited, [:CreditScore])

predict(treem, a)
get_features(treem)
feature_importance(treem, a)

jlt = trees(treem)[1]
df = a
loss = LogitLogLoss()
features = setdiff(names(a), [:Exited])
target = :Exited
warmstart = fill(0, length(nrow(a)))
verbose = false
lambda = 0
gamma = 0


rows_bool = fill(true, nrow(df))
freq_dict = Dict{Symbol, Int}()
gain_dict = Dict{Symbol, Float64}()
coverage_dict = Dict{Symbol, Float64}()

Gs = JLBoost.g.(loss, df[!, target], jlt.weight)
Hs = JLBoost.h.(loss, df[!, target], jlt.weight)



if !isequal(jlt.splitfeature, missing)

!isequal(jlt.splitfeature, missing)
# compute the Quality/Gain. Coverage
rows_bool_left = rows_bool .& (df[!, jlt.splitfeature] .<= jlt.split)

rows_bool_right = rows_bool .& (.!rows_bool_left)

G_left = sum(@view(Gs[rows_bool_left]))
H_left = sum(@view(Hs[rows_bool_left]))

G_right = sum(@view(Gs[rows_bool_right]))
H_right = sum(@view(Hs[rows_bool_right]))



# note that hyper parameters are not used to compute the gain
gain = (H_left == 0 ? 0 : G_left^2/H_left) + (H_right == 0 ? 0 : G_right^2/H_right) - (G_left + G_right)^2/(H_left + H_right)
coverage = H_left + H_right

if haskey(freq_dict, jlt.splitfeature)
    freq_dict[jlt.splitfeature] += 1
    gain_dict[jlt.splitfeature] += gain
    coverage_dict[jlt.splitfeature] += coverage
else
    freq_dict[jlt.splitfeature] = 1
    gain_dict[jlt.splitfeature] = gain
    coverage_dict[jlt.splitfeature] = coverage
end
print(gain_dict)

JLBoost.feature_importance!(jlt.children[1], df, loss, target, rows_bool_left,  freq_dict, gain_dict, coverage_dict, Gs, Hs)
JLBoost.feature_importance!(jlt.children[2], df, loss, target, rows_bool_right, freq_dict, gain_dict, coverage_dict, Gs, Hs)
end

################################
jlt = jlt.children[1]
rows_bool = rows_bool_left

jlt = jlt.children[2]
rows_bool = rows_bool_right
################################


(freq_dict = freq_dict, gain_dict = gain_dict, coverage_dict = coverage_dict)
end

using MLJ

model = JLBoostModel()

X, y = unpack(a, !=(:Exited), ==(:Exited))

mljmodel = fit(model, 0, X, y)
predict(model, mljmodel.fitresult, X)

fitted_params(model, mljmodel.fitresult)
