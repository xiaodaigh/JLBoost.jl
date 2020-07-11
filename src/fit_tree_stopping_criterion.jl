module TreeFitStoppingCriterions

export max_depth

function max_depth(depth)
    function (jlt::AbstractJLBoostTree)
        nodedepth(jlt) >= depth
    end
end

end