export depth_wise, lossguide

using ..JLBoostTrees: AbstractJLBoostTree, get_leaf_nodes

# Note the naming conventions here come from xgboost
# see https://xgboost.readthedocs.io/en/latest/parameter.html -- grow_policy


"""
    A function that select leaf-node for further growth in a balanced manner, i.e. left and right
    at the same time

* jlt - A JLBoostTree
"""
function depth_wise(jlt::AbstractJLBoostTree)
    leaf_nodes = filter(x -> !ismissing(x.gain) && (x.gain > 0), get_leaf_nodes(jlt))

    return leaf_nodes
end

function lossguide(jlt::AbstractJLBoostTree)::Vector{JLBoostTree}
    leaf_nodes = filter(x->!ismissing(x.gain), get_leaf_nodes(jlt))
    leaf_nodes = filter(x->x.gain > 0, leaf_nodes)

    if length(leaf_nodes) == 0
        return JLBoostTree[]
    else
        # the gains are stored in the parents
        _, pos = findmax(map(x->x.gain, leaf_nodes))
        return [leaf_nodes[pos]]
    end
end