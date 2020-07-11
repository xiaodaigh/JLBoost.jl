# Column sample strategy
# At many points in the course of being a tree model with JLBoost you will need to choose a set of
# columns this defines the strategy to choose them

module ColumnSampleStrategy

using ..JLBoostTrees: AbstractJLBoostTree
export ColumnSimpleRandomSample, ColumnNoSample, SameColumnAsParentSample

import StatsBase: sample

abstract type AbstractColumnSampleStrategy end

# performs simple random sampling i.e. every column has the same chance of being selected
struct ColumnSimpleRandomSample <: AbstractColumnSampleStrategy
    frac # the fraction of columns to sample
end
"""
    sample(S::AbstractColumnSampleStrategy, cols, current_node::AbstractJLBoostTree)

* S - The sampling strategy
* features - The iterable of possible features to choose from
* current_node - The `current_node` in the tree that needs to select the node
"""
function sample(S::ColumnSimpleRandomSample, features, _...)
    num_of_features_to_sample = ceil(Int, length(features)*S.frac)
    sample(features, num_of_features_to_sample; replace = true)
end

# do no sampling at all
struct ColumnNoSample <: AbstractColumnSampleStrategy end

function sample(::ColumnNoSample, features, _...)
    features
end

# sample only one feature, the one that has been used to split in the parent
# this can be very useful as the data is already sorted by this column so it will be
# quite fast
struct SameColumnAsParentSample <: AbstractColumnSampleStrategy end

function sample(::SameColumnAsParentSample, features, current_node::AbstractJLBoostTree, _...)
    (current_node.parent.splitfeature, )
end




end