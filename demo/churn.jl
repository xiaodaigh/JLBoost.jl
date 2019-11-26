using Pkg
Pkg.activate(".")
@time using JLBoost
@time using DataFrames
@time using JDF, LossFunctions
using DataConvenience: replace_onehot!

###############################################################################
# testing best_split
###############################################################################
if !isdir("c:/data/Churn_Modelling_fnl_w_profit.jdf")
    using CSV
    a = CSV.read("c:/data/Churn_Modelling_fnl_w_profit.csv")
    savejdf(a, "c:/data/Churn_Modelling_fnl_w_profit.jdf")
end

@time a = loadjdf("c:/data/Churn_Modelling_fnl_w_profit.jdf")
a[!, :Geography] = categorical(a[!, :Geography])
a[!, :Gender] = categorical(a[!, :Gender])

replace_onehot!(a, :Geography)
replace_onehot!(a, :Gender)

@time m = jlboost(a, :Exited, setdiff(names(a), [:Exited, :hist_mthly_profit]))
feature_importance(m, a)

if false # for debuggin
    jlt = trees(treem)[1]
    df = a
    loss = LogitLogLoss()
    features = setdiff(names(a), [:Exited])
    target = :Exited
    warmstart = fill(0, length(nrow(a)))
    verbose = false
    lambda = 0
    gamma = 0
end


using MLJ

model = JLBoostModel()

X, y = unpack(a, !=(:Exited), ==(:Exited))

mljmodel = fit(model, 0, X, y)
predict(model, mljmodel.fitresult, X)

fitted_params(model, mljmodel.fitresult)
