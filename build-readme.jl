# Weave readme
using Pkg
Pkg.activate("readme-env")
Pkg.update()

using PkgVersionHelper: upcheck
upcheck()

if false
    Pkg.add("Weave")
    Pkg.add("RDatasets")
    Pkg.add("DataFrames")
    Pkg.add("JDF") # needed for table check
    Pkg.add("LossFunctions")
    Pkg.update()
end

using Weave

weave("README.jmd", out_path=:pwd, doctype="github")

# output to README.jl for easy testing
if false
    tangle("README.jmd")
    # creates a  file
    "README.jl"
end
