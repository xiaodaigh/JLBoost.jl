module JLBoostTrees

import Base: show, *, print, println, +, getproperty
import Base: vcat

export JLBoostTree, JLBoostTreeModel, show, trees
export WeightedJLBoostTree, *, AbstractJLBoostTree, print, println, vcat
export getproperty

# this needs to be defined before everything else
include("tree-and-tree-models.jl")

include("abstract-tree-interface.jl")

include("children.jl")
include("filter_tbl_by_splits.jl")
include("get_leaf_nodes.jl")
include("traits.jl")
include("treedepth.jl")

end # end module
