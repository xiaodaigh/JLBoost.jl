# Weave readme
using Pkg
cd("c:/git/JLBoost")
Pkg.activate("c:/git/JLBoost")

Pkg.add("Weave")
Pkg.add("RDatasets")
Pkg.add("JDF") # needed for table check
Pkg.update()
#Pkg.add("MLJ")  needed for table check
#Pkg.add("JLBoostMLJ") # needed for table check

using Weave

weave("README.jmd", out_path=:pwd, doctype="github")

Pkg.rm("Weave")
Pkg.rm("RDatasets")
Pkg.rm("JDF") # needed for table check
#Pkg.rm("MLJ")
#Pkg.rm("JLBoostMLJ")

# output to README.jl for easy testing
tangle("README.jmd")
