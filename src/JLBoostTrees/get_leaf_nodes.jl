export get_leaf_nodes, get_leaf_nodes!

"""
    get_leaf_nodes(jlt::AbstractJLBoostTreeModel)

jlt - The JLBoostTree


"""
function get_leaf_nodes(jlt::AbstractJLBoostTree)
    T = eltype(jlt.children)
    get_leaf_nodes!(T[], jlt)
end

function get_leaf_nodes!(leaf_nodes::AbstractVector{T}, jlt::AbstractJLBoostTree) where {T <: AbstractJLBoostTree}
    if has_children(jlt)
        for child in children(jlt)
            get_leaf_nodes!(leaf_nodes, child)
        end
    else
        push!(leaf_nodes, jlt)
    end
    leaf_nodes
end
