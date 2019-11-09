module JLBoostTrees

using DataFrames
import Base: show

mutable struct JLBoostTreeNode{T <: AbstractFloat}
    weight::T
    children::Vector{JLBoostTreeNode}
    splitfeature
    split
    JLBoostTreeNode(w::T) where {T <: AbstractFloat}  = new{T}(w, JLBoostTreeNode[], missing, missing)
end

mutable struct JLBoostTree
    parentnode::JLBoostTreeNode
    df::AbstractDataFrame
    target::Symbol
    features::Vector{Symbol}
    prev_w::Symbol # the column that represents the sum of previous weights 
    eta::Real
    lambda::Real
    gamma::Real
    maxdepth::Int
    subsample::Real
end

function showlah(jlt::JLBoostTreeNode, ntabs::I; splitfeature="") where {I <: Integer}
    if ntabs == 0
        tabs = ""
    else ntabs >= 1
        tabs = reduce(*, ["  " for i = 1:ntabs])
    end
    if splitfeature != ""
        println("")
        print("$tabs -- $splitfeature")
        if ismissing(jlt.splitfeature)
            println("")
            println("  $tabs ---- weight = $(jlt.weight)")
        else
            #println("  $tabs ---- $(jlt.splitfeature), weight = $(jlt.weight)")
            #println("  $tabs ---- $(jlt.splitfeature)")
        end
    elseif ismissing(jlt.splitfeature)
        println("$tabs weight = $(jlt.weight)")
    else
        #println("$tabs $(jlt.splitfeature), weight = $(jlt.weight)")
    end

    if length(jlt.children) == 2
        showlah(jlt.children[1], ntabs + 1; splitfeature = "$(jlt.splitfeature) <= $(jlt.split)")
        showlah(jlt.children[2], ntabs + 1; splitfeature = "$(jlt.splitfeature) > $(jlt.split)")
    end
end

function show(io::IO, jlt::JLBoostTreeNode)
    showlah(jlt,0)
end
end
