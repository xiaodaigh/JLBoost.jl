# This file implements the interface of AbstractTrees.jl
# https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Abstract-Tree-Interface

# AbstractTrees.children in children.jl
# AbstractTrees.nodevalue AbstractTrees.nodevalue(AbstractJLBoostTree)

import AbstractTrees

using Tables: Tables;

# JLBoostTrees have parents stored
AbstractTrees.ParentLinks(::AbstractJLBoostTree) = AbstractTrees.StoreParent()

AbstractTrees.SiblingLinks(::AbstractJLBoostTree) = AbstractTrees.ImplicitSiblings()

AbstractTrees.ChildIndexing(::AbstractJLBoostTree) = AbstractTrees.IndedChildren()

AbstractTrees.NodeType(::AbstractJLBoostTree) = AbstractTrees.HasNodeType()

struct FeatureSplitPredictate
    feature
    split_val
    inclusive::Bool
end

(f::FeatureSplitPredictate)(tbl) = begin
    col = Tables.getcolumn(tbl, f.feature)
    col .< f.split_val
end

## return a predictate (function) that when applied to a Tables.jl table can return a vector of
# indices for which children to choose. The indices for binary trees are usually `true` and `false`
# for the left and right children respectively.
AbstractTrees.nodevalue(jlt::AbstractJLBoostTree) = begin
    #TODO "implement inclusive"
    FeatureSplitPredictate(jlt.splitfeature, jlt.split, true)
end

AbstractTrees.StableNode(::AbstractJLBoostTree) = NodeTypeUnknonw()

AbstractTrees.ischild(jlt::AbstractJLBoostTree) = !isnothing(jlt.parent)
AbstractTrees.isroot(jlt::AbstractJLBoostTree) = isnothing(jlt.parent)

# NOT implemented interfaces include
# * AbstractTrees.getdescendants
# * AbstractTrees.nodevalues
# * AbstractTrees.ischild
# * AbstractTrees.isdescendent
# * AbstractTrees.treebreadth
# * AbstractTrees.treeheight
# * AbstractTrees.desecendleft
# * AbstractTrees.getroot
