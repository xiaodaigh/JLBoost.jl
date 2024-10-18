using Pkg
using DataFrames
using JDF
using JLBoost
using StatsBase: mode

###############################################################################
# fitting GiveMeSomeCredit
###############################################################################
if !isdir("c:/data/GiveMeSomeCredit/cs-training.jdf")
    using CSV
    a = CSV.read("c:/data/GiveMeSomeCredit/cs-training.csv", missingstring=["NA"], DataFrame)
    # create_missing!(a, :MonthlyIncome)
    # create_missing!(a, :NumberOfDependents)
    type_compress!(a, compress_float=true)
    savejdf("c:/data/GiveMeSomeCredit/cs-training.jdf", a)
end

###############################################################################
# testing find_best_split
###############################################################################
@time a = loadjdf("c:/data/GiveMeSomeCredit/cs-training.jdf")

a = DataFrame(a)

for colname in names(a)
    println(colname)

    c = count(ismissing, a[!, colname])

    c == 0 ? continue : nothing

    # m = sum(a[!, colname] |> skipmissing)/c
    m = mode(a[!, colname] |> skipmissing)
    rows = ismissing.(a[!, colname])
    a[rows, colname] .= m
end


features = setdiff(names(a), ["SeriousDlqin2yrs"])
target = :SeriousDlqin2yrs
verbose = false

m = jlboost(a, target)

m(a)

predict(m, a)

x = JLBoost.AUC_plot_data(-m(a), a[!, target])

using Plots: plot

plot(x)
