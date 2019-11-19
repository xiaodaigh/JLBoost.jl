using TraitWrappers

struct IterableTraitWrapper{T} <: AbstractTraitWrapper
    object::T
    IterableTraitWrapper(t::T) = new{T}(t)
end

struct ColumnBangAccessible{T} <: AbstractTraitWrapper
    object::T
    ColumnBangAccessible(t::T) = new{T}(t)
end
