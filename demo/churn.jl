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

@time treem = jlboost(a, :Exited)

predict(treem, a)
get_features(treem)
feature_importance(treem, a)

using MLJ

model = JLBoostModel()

X, y = unpack(a, !=(:Exited), ==(:Exited))

mljmodel = fit(model, 0, X, y)
predict(model, mljmodel.fitresult, X)

fitted_params(model, mljmodel.fitresult)
