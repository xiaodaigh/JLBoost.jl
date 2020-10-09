# using TraitWrappers

# abstract type AbstractJLBoostPredictTraitWrapper <: AbstractTraitWrapper end
#
# struct JLBoostHasPredictTraitWrapper <: AbstractJLBoostPredictTraitWrapper
#     object::T
# end
#
# struct IterableTraitWrapper{T} <: AbstractTraitWrapper
#     object::T
# end
#
# struct ColumnBangAccessible{T} <: AbstractTraitWrapper
#     object::T
#     ColumnBangAccessible(t::T) = new{T}(t)
# end
#
# # checks whether T satisfies the IterableTraitWrapper
# satisfy(::Type{T}) where T = begin
#     hasmethod(iterate, Tuple{T})
# end
#
# satisfy(::Type{W}) where W <: AbstractVector = true
#
# abstract type ABC end
#
# ABC(x) = begin
#     satisfy(ABC, x)
# end
#
# satisfy(::Type{<:ABC}, obj) = begin
#     println("ok")
#     true
# end
#
# struct DEF <: ABC
#     object
# end
#
# DEF(1)
#
# satisfy(DEF, 1)
