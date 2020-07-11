export children, has_children

import AbstractTrees: children, has_children

function children(jlt::AbstractJLBoostTree)
    jlt.children
end

function has_children(jlt::AbstractJLBoostTree)
    length(children(jlt)) > 0
end