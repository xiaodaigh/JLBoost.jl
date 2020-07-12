module TreeFitStoppingCriterion

 using ..JLBoostTrees: AbstractJLBoostTree

export max_depth

function max_depth(depth)
    function (jlt::AbstractJLBoostTree)
        treedepth(jlt) >= depth
    end
end

end