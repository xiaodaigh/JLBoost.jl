export treedepth

"""
    treedepth(jlt::AbstractJLBoostTree)

Return the depth of the `jlt` tree
"""
function treedepth(jlt::AbstractJLBoostTree)
    if has_children(jlt)
        return 1 + maximum(treedepth, children(jlt))
    else
        return 0
    end
end