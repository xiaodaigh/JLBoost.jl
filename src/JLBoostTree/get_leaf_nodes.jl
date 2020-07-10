export get_leaf_nodes, get_leaf_nodes!

"""
    get_leaf_nodes(jlt::AbstractJLBoostTreeModel)

jlt - The JLBoostTree


"""
function get_leaf_nodes(jlt::AbstractJLBoostTree)
    get_leaf_nodes!(AbstractJLBoostTree[], jlt)
end

function get_leaf_nodes!(leaf_nodes::Vector{AbstractJLBoostTree}, jlt::AbstractJLBoostTree)
    if length(jlt.children) == 0
        push!(leaf_nodes, jlt)
    else
        for child in jlt.children
            get_leaf_nodes!(leaf_nodes, child)
        end
    end
    leaf_nodes
end
