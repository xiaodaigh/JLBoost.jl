# Weave readme
using Pkg
Pkg.activate(".")

Pkg.add("Weave")
Pkg.add("RDatasets")
Pkg.add("MLJ")
Pkg.add("JLBoostMLJ")
using Weave



weave("README.jmd", out_path=:pwd, doctype="github")

Pkg.rm("Weave")
Pkg.rm("RDatasets")
Pkg.rm("MLJ")
Pkg.rm("JLBoostMLJ")


# output to README.jl for easy testing
# tangle("README.jmd")
