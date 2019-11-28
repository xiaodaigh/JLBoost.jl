using Pkg
Pkg.activate(".")
@time using DataFrames
@time using JDF
@time using JLBoost, LossFunctions

###############################################################################
# testing best_split
###############################################################################
@time a = loadjdf("c:/data/GiveMeSomeCredit/cs-training.jdf")

loss = LogitLogLoss()
features = setdiff(names(a), [:SeriousDlqin2yrs, :column1])
target = :SeriousDlqin2yrs
warmstart = fill(0, length(a.age))
verbose = false
lambda = 0
gamma = 0

a[!, :SeriousDlqin2yrs] = allowmissing(a[!, :SeriousDlqin2yrs])

#a[!, :DebtRatio] = allowmissing(a[!, :DebtRatio])
#a[rand(1:nrow(a), 15_000), :DebtRatio] .= missing

@time treem = jlboost(a, target, setdiff(features, [:NumberOfTimes90DaysLate]))

predict(treem, a)
get_features(treem)
feature_importance(treem, a)

using MLJ

model = JLBoostClassifier()

X, y = unpack(a, !=(:SeriousDlqin2yrs), ==(:SeriousDlqin2yrs))

mljmodel = fit(model, 0, X, y)
predict(model, mljmodel.fitresult, X)

fitted_params(model, mljmodel.fitresult)
