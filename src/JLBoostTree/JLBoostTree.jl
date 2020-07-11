module JLBoostTrees

import Base: show, *, print, println, +, getproperty
import Base: vcat

export JLBoostTree, JLBoostTreeModel, show, trees
export WeightedJLBoostTree, *, AbstractJLBoostTree, print, println, vcat
export getproperty

include("tree-and-tree-models.jl")
include("children.jl")
include("get_leaf_nodes.jl")
include("traits.jl")

end # end module
