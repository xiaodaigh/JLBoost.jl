# Note the naming conventions here come from xgboost
# see https://xgboost.readthedocs.io/en/latest/parameter.html -- grow_policy


export depth_wise

"""
    A function that select leaf-node for further growth in a balanced manner, i.e. left and right
    at the same time

* jlt - A JLBoostTree
"""
function depth_wise(jlt::AbstractJLBoostTree)
    # TODO stub atm
    warn("add the selection function properly")
    return jlt
end

function lossguide(jlt::AbstractJLBoostTree)
    warn("add the selection function properly")
    return jlt
end