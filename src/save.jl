#export save, load

using Serialization: serialize, deserialize

save(jlt::T, file::AbstractString) where {T<:Union{JLBoostTreeModel, JLBoostTree, Vector{JLBoostTree}}} = begin
    open(file, "w") do io
        serialize(io, jlt)
    end
end

load(file::AbstractString)::Union{JLBoostTreeModel, JLBoostTree, Vector{JLBoostTree}} = begin
    open(file, "r") do io
        deserialize(io)
    end
end
