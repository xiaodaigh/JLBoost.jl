using Pkg
#Pkg.activate("c:/git/JLBoost")
@time using DataFrames
@time using JDF
@time using JLBoost, LossFunctions

###############################################################################
# fitting GiveMeSomeCredit
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

###############################################################################
# testing find_best_split
###############################################################################
@time a = loadjdf("c:/data/GiveMeSomeCredit/cs-training.jdf")

loss = LogitLogloss()

features = setdiff(names(a), [:SeriousDlqin2yrs])
target = :SeriousDlqin2yrs
warmstart = fill(0, length(a.age))
verbose = false
lambda = 0
gamma = 0

using ForwardDiff: derivative

f(warmstart, target) = begin
	p = 1/(1+exp(-(warmstart)))
	-(target*log(p) + (1-target)*log(1-p))
end


f(target) = warmstart -> f(warmstart, target)

x = derivative.(f.(a.SeriousDlqin2yrs), fill(0.0, nrow(a)))

x1 = deriv.(loss, a.SeriousDlqin2yrs, fill(0.0, nrow(a)))

x == x1
