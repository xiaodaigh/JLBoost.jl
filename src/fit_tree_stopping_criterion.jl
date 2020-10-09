module TreeFitStoppingCriterion

export max_depth_stopping_criterion, max_leaves_stopping_criterion

using ..JLBoostTrees: AbstractJLBoostTree, treedepth, get_leaf_nodes



function max_depth_stopping_criterion(depth)
    function (jlt::AbstractJLBoostTree)
        treedepth(jlt) >= depth
    end
end

function max_leaves_stopping_criterion(leaves)
    function (jlt::AbstractJLBoostTree)
        length(get_leaf_nodes(jlt)) >= leaves
    end
end

end