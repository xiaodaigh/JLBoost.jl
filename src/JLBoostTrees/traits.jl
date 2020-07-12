export nodetype, parentlinks

import AbstractTrees: nodetype, ParentLinks, StoredSiblings

function nodetype(::T) where {T <: AbstractJLBoostTree}
    T
end

function parentlinks(::T) where {T <: AbstractJLBoostTree}
    AbstractTrees.StoredParents()
end