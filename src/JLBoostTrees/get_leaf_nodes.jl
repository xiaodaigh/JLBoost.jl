export get_leaf_nodes, get_leaf_nodes!

"""
    get_leaf_nodes(jlt::AbstractJLBoostTreeModel)

jlt - The JLBoostTree


"""
function get_leaf_nodes(jlt::AbstractJLBoostTree)
    get_leaf_nodes!(AbstractJLBoostTree[], jlt)
end

function get_leaf_nodes!(leaf_nodes::Vector{AbstractJLBoostTree}, jlt::AbstractJLBoostTree)
    if has_children(jlt)
        for child in children(jlt)
            get_leaf_nodes!(leaf_nodes, child)
        end
    else
        push!(leaf_nodes, jlt)
    end
    leaf_nodes
end
