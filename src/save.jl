#export save, load

using Serialization: serialize, deserialize

save(
    jlt::T,
    file::AbstractString,
) where {T<:Union{JLBoostTreeModel,JLBoostTree,Vector{<:AbstractJLBoostTree}}} = begin
    print("testing save")
    open(file, "w") do io
        serialize(io, jlt)
    end
end

load(
    file::AbstractString,
)::Union{JLBoostTreeModel,JLBoostTree,Vector{<:AbstractJLBoostTree}} = begin
    open(file, "r") do io
        deserialize(io)
    end
end
