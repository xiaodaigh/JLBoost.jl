# Weave readme
using Pkg
Pkg.activate(".")
using Weave

weave("README.jmd", out_path=:pwd, doctype="github")
