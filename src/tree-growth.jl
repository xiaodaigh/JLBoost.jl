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
    # TODO stub atm
    return get_leaf_nodes(jlt)
end

function lossguide(jlt::AbstractJLBoostTree)
    error("STUB: add the selection function properly")
    return jlt
end