using Pkg
Pkg.activate("c:/git/JLBoost")
@time using DataFrames
@time using JDF
@time using JLBoost, LossFunctions


###############################################################################
# fitting fm
###############################################################################
if !isdir("c:/data/GiveMeSomeCredit/cs-training.jdf")
	using CSV
	a = CSV.read("c:/data/GiveMeSomeCredit/cs-training.csv", missingstrings=["NA"])
	rename!(a, Symbol("")=>:column1)
	create_missing!(a, :MonthlyIncome)
	create_missing!(a, :NumberOfDependents)
	type_compress!(a, compress_float=true)
	savejdf("c:/data/GiveMeSomeCredit/cs-training.jdf", a)
end

a = loadjdf("c:/data/GiveMeSomeCredit/cs-training.jdf")

using Calculus

softmax(w) = 1/(1 + exp(-w))
logloss(y) = w -> logloss(y, w)
logloss(y, w) = -(y*log( 1/(1 + exp(-w))) + (1-y)*log(1- 1/(1 + exp(-w))))

t = a.SeriousDlqin2yrs
ws = fill(0.0,nrow(a))

using CuArrays
ct = cu(t)
cws = cu(ws)

logloss.(ct, cws)

ct .+ cws

using BenchmarkTools
@benchmark Calculus.derivative.(logloss.($t), $ws)

using LossFunctions: deriv
loss = LogitLogLoss()
@benchmark deriv(loss, $t, $ws)

using ForwardDiff
@benchmark ForwardDiff.derivative.(logloss.($t), $ws)

using ForwardDiff

d = transpose(hcat(t, ws))
@benchmark ForwardDiff.gradient.(x->logloss(x...), eachcol(d))

using Zygote: @adjoint, gradient

@adjoint logloss(y, w) = logloss(y, w), Δ -> (Δ*-(log(1/(1+exp(-w))) - (log(1 - 1/(1+exp(-w))))), Δ*( -1/(1+exp(w)) - y + 1))

@time gradient(logloss, 1, 0.5)

f(t, ws) = gradient(logloss, t, ws)[2]

@benchmark f.(t, ws)

# Winner is LossFunctions

using CuArrays
ct = cu(t)
cws = cu(ws)

g(ct, cws) = deriv(LogitLogloss(), ct, cws)

@benchmark g.(loss, ct, cws)

#####################################################################

using Zygote: @adjoint, gradient
gg(ct, cws) = gradient(logloss, ct, cws)[2]
@time gg.(ct, cws)

logloss.(ct, cws)

|
d1(y,w) = @. -1 / (1 + exp(w)) - y + 1

using BenchmarkTools
CuArrays.allowscalar(false)
@benchmark d1(ct, cws)
