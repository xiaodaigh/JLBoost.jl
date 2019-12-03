module JLBoostTrees

import Base: show, *, print, println, +

export JLBoostTree, JLBoostTreeModel, show, trees
export WeightedJLBoostTree, *, AbstractJLBoostTree, print, println, vcat

abstract type AbstractJLBoostTree end

mutable struct JLBoostTreeModel
	jlt::Vector
	loss
	target::Symbol
end

"""
	trees(jlt::JLBoostTreeModel)

Return the tree
"""
trees(jlt::JLBoostTreeModel) = jlt.jlt

import Base: vcat

+(v1::JLBoostTreeModel, v2::JLBoostTreeModel) = begin
	v3 = deepcopy(v1)
	v3.jlt = vcat(trees(v1), trees(v2))
	v3
end

mutable struct JLBoostTree <: AbstractJLBoostTree
    weight
	parent::Union{JLBoostTree, Nothing}
    children::Vector
    splitfeature
    split
    JLBoostTree(w::T) where {T <: AbstractFloat}  = new(w, nothing, JLBoostTree[], missing, missing)
	JLBoostTree(w::T, parent::JLBoostTree) where {T <: AbstractFloat}  = new(w, parent, JLBoostTree[], missing, missing)
end

mutable struct WeightedJLBoostTree <: AbstractJLBoostTree
	tree::JLBoostTree
	eta::Number
end

getproperty(jlt::WeightedJLBoostTree, sym::Symbol) = begin
	if sym == :eta
		return getfield(jlt, :eta)
	elseif sym == :tree
		return getfield(jlt, :tree)
	else
		return getproperty(getfield(jlt, :tree), sym)
	end
end

*(jlt::JLBoostTree, eta::Number) = WeightedJLBoostTree(jlt, eta)

*(eta::Number, jlt::JLBoostTree) = WeightedJLBoostTree(jlt, eta)

*(jlt::WeightedJLBoostTree, eta::Number) = begin
	jlt.eta *= eta
	jlt
end

*(eta::Number, jlt::WeightedJLBoostTree) = begin
	jlt.eta *= eta
	jlt
end


"""
	show(io, jlt, ntabs; splitfeature="")

Show a JLBoostTree
"""
function show(io, jlt::WeightedJLBoostTree, ntabs::I; kwargs...) where {I <: Integer}
	println(io, "eta = $(jlt.eta) (tree weight)")
	show(io, jlt.tree, ntabs;  kwargs...)
end


function show(io, jlt::JLBoostTree, ntabs::I; splitfeature="") where {I <: Integer}
    if ntabs == 0
        tabs = ""
    else ntabs >= 1
        tabs = reduce(*, ["  " for i = 1:ntabs])
    end
    if splitfeature != ""
        println(io, "")
        print(io, "$tabs -- $splitfeature")
        if ismissing(jlt.splitfeature)
            println(io, "")
            println(io, "  $tabs ---- weight = $(jlt.weight)")
        else
            #println(io, "  $tabs ---- $(jlt.splitfeature), weight = $(jlt.weight)")
            #println(io, "  $tabs ---- $(jlt.splitfeature)")
        end
    elseif ismissing(jlt.splitfeature)
        println(io, "$tabs weight = $(jlt.weight)")
    else
        #println("$tabs $(jlt.splitfeature), weight = $(jlt.weight)")
    end

    if length(jlt.children) == 2
        show(io, jlt.children[1], ntabs + 1; splitfeature = "$(jlt.splitfeature) <= $(jlt.split)")
        show(io, jlt.children[2], ntabs + 1; splitfeature = "$(jlt.splitfeature) > $(jlt.split)")
    end
end

"""
	show(io, jlt)

Show a JLBoostTree or a Vector{JLBoostTree}
"""
function show(io::IO, jlt::AbstractJLBoostTree)
    show(io, jlt, 0)
end

function show(io::IO, ::MIME"text/plain", jlt::AbstractVector{T}) where T <: AbstractJLBoostTree
	for (i, tree) in enumerate(jlt)
		println(io, "Tree $i")
    	show(io, tree, 0)
    	println(io, " ")
    end
end

end # end module
