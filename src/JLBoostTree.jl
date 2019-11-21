module JLBoostTrees

using DataFrames
import Base: show, *, print, println

export JLBoostTree, show
export WeightedJLBoostTree, *, AbstractJLBoostTree, print, println

abstract type AbstractJLBoostTree end

mutable struct JLBoostTree{T <: AbstractFloat} <: AbstractJLBoostTree
    weight::T
	parent::Union{JLBoostTree{T}, Nothing}
    children::Vector{JLBoostTree{T}}
    splitfeature
    split
    JLBoostTree(w::T) where {T <: AbstractFloat}  = new{T}(w, nothing, JLBoostTree{T}[], missing, missing)
	JLBoostTree(w::T, parent::JLBoostTree{T}) where {T <: AbstractFloat}  = new{T}(w, parent, JLBoostTree{T}[], missing, missing)
end

mutable struct WeightedJLBoostTree{T <:AbstractFloat, W<:Number} <: AbstractJLBoostTree
	tree::JLBoostTree{T}
	eta::W
end

*(jlt::JLBoostTree, eta::Number) = WeightedJLBoostTree(jlt, eta)

*(eta::Number, jlt::JLBoostTree) = WeightedJLBoostTree(jlt, eta)


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
