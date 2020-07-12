export children, has_children, is_left_child, is_right_child

import AbstractTrees: children, has_children

function children(jlt::AbstractJLBoostTree)
    jlt.children
end

function has_children(jlt::AbstractJLBoostTree)
    length(children(jlt)) > 0
end

function is_left_child(jlt::AbstractJLBoostTree)
    jlt == jlt.parent.children[1]
end

is_right_child(jlt::AbstractJLBoostTree) = !is_left_child(jlt)