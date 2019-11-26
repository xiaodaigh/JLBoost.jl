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
a[!, :Geography] = categorical(a[!, :Geography])
a[!, :Gender] = categorical(a[!, :Gender])

replace_onehot!(a, :Geography)
replace_onehot!(a, :Gender)

@time m = jlboost(a, :Exited, setdiff(names(a), [:Exited, :hist_mthly_profit]))
feature_importance(m, a)

using TableView
showtable(a)

using BenchmarkTools



a[!, :dummy] .= 1
a[!, :row_id] .= 1:nrow(a)

unstack(a[!, [:row_id, :Geography, :dummy]], :Geography, :dummy)

x = a[!, :Geography]

a[!, :Geography] = categorical(a[!, :Geography])

using Flux: onehotbatch
using DataAPI: refarray, refvalue, defaultarray, refpool

x = a[!, :Geography]
x.refs
x.pool.index
fieldnames(typeof(x.pool))

xx = onehotbatch(x.refs, 1:length(x.pool.index))

a[!, :Geography_Germany] = xx[1, :]



(a[!, :Geography])
refarray(a[!, :Geography])
refvalue(a[!, :Geography])

refpool(a[!, :Geography])

@time treem = jlboost(a, :Exited, [:Age])

@time treem2 = jlboost(a, :Exited, [:CreditScore], predict(treem, a))

treem3 = treem + treem2


AUC(-predict(treem, a), a.Exited)

AUC(-predict(treem2, a), a.Exited)

AUC(-predict(treem3, a), a.Exited)


@time treem = jlboost(a, :Exited, [:Age, :CreditScore])

predict(treem, a)
get_features(treem)
feature_importance(treem, a)

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
