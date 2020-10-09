# Weave readme
using Pkg
cd("c:/git/JLBoost")
Pkg.activate("c:/git/JLBoost/jlboost-test")

if false
    Pkg.add("Weave")
    Pkg.add("RDatasets")
    Pkg.add("DataFrames")
    Pkg.add("JDF") # needed for table check
    Pkg.add("LossFunctions")
    Pkg.update()
    #Pkg.add("MLJ")  needed for table check
    #Pkg.add("JLBoostMLJ") # needed for table check
end

using Weave

weave("c:/git/JLBoost/README.jmd", out_path = :pwd, doctype = "github")

# Pkg.rm("Weave")
# Pkg.rm("RDatasets")
# Pkg.rm("JDF") # needed for table check
#Pkg.rm("MLJ")
#Pkg.rm("JLBoostMLJ")

# output to README.jl for easy testing
if false
    tangle("c:/git/JLBoost/README.jmd")
end
