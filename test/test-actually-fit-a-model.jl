using Pkg
Pkg.activate("c:/git/JLBoost")
@time using JLBoost
@time using DataFrames, JDF


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

@time jlboost(a, :SeriousDlqin2yrs)

@time jlboost(a, :SeriousDlqin2yrs, max_leaves = 32)
