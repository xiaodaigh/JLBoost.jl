# Weave readme
using Pkg
Pkg.activate(".")

Pkg.add("Weave")
Pkg.add("RDatasets")
Pkg.add("JDF") # needed for table check
Pkg.add("MLJ") # needed for table check
#Pkg.add("JLBoostMLJ") # needed for table check


# using Pkg
#
# Pkg.add("Tables")
#
# using JLBoost, RDatasets, JDF
# iris = dataset("datasets", "iris")
#
# iris[!, :is_setosa] = iris[!, :Species] .== "setosa"
# target = :is_setosa
#
# features = setdiff(names(iris), [:Species, :is_setosa])
#
# savejdf("iris.jdf", iris)
# irisdisk = JDFFile("iris.jdf")
#
# using Tables
#
# Tables.istable(irisdisk)

# fit using on disk JDF format
# xgtree1 = jlboost(irisdisk, target, features)

using Weave

weave("README.jmd", out_path=:pwd, doctype="github")

Pkg.rm("Weave")
Pkg.rm("RDatasets")
Pkg.rm("JDF") # needed for table check
Pkg.rm("MLJ")
#Pkg.rm("JLBoostMLJ")

# output to README.jl for easy testing
# tangle("README.jmd")
