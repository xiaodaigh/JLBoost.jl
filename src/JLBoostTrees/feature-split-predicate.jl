# A feature split predictate is a function that takes a Tables.jl table and
# returns a boolean vector of the same length as the number of rows in the table. The boolean vector
# is true if the row should be kept and false if it should be dropped.

using Missings: ismissing

# an abstract type for feature split predictates
abstract type AbstractFeatureSplitPredictate end

struct FeatureSplitPredictate <: AbstractFeatureSplitPredictate
    feature
    split_val
end

(f::FeatureSplitPredictate)(tbl) = begin
    col = Tables.getcolumn(tbl, f.feature)
    col .< f.split_val
end

struct MissingWrapperPredicate <: AbstractFeatureSplitPredictate
    feature
    split_val
    missing_go_left::Bool
end

(f::MissingWrapperPredicate)(tbl) = begin
    col = Tables.getcolumn(tbl, f.feature)

    if missing_go_left
        return ismissing.(col) .|| col .<= f.split_val
    else
        return .!ismissing.(col) .&& col .<= f.split_val
    end
end