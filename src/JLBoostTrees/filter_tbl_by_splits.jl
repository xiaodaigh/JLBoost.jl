export keeprow_vec, filter_tbl_by_splits

using DataFrames: nrow
using Tables

"""
    Returns the boolean that can used to filter the table

* tbl - A Tables.jl table
* node - A JLBoostTree
"""
function keeprow_vec(tbl, node::Union{Nothing, AbstractJLBoostTree})::BitArray
    @assert Tables.istable(tbl)

    tblc = Tables.columns(tbl)

    # a boolean on whether to keep the row
    if node === nothing
        return keeprow = trues(nrow(tbl))
    end

    if node.parent.parent === nothing
        keeprow = trues(nrow(tbl))
    else
        keeprow = keeprow_vec(tbl, node.parent)
    end

    tmp = getproperty(tblc, node.parent.splitfeature)
    if is_left_child(node)
        # TODO fix this
        # @warn "Currently assuming missing go left; this NEEDS to be FIXED"
        keeprow .&= ismissing.(tmp) .|| tmp .<= node.parent.split
        # keeprow .&= tmp .<= node.parent.split
    else
        # @warn "Currently assuming missing go left; this NEEDS to be FIXED"
        # keeprow .&= .!ismissing.(tmp) .|| tmp .> node.parent.split
        # the above causes errors so apply De Morgan's law
        # not(missing) | x > y = not not (not(missing) | x > y)
        #                      = not (missing | x <= y) # by De Morgan's law
        # keeping the above as reminder of how dumb I am. Just negate man
        keeprow .&= .!(ismissing.(tmp) .|| tmp .<= node.parent.split)
        # keeprow .&= tmp .> node.parent.split
    end

    keeprow
end

"""
    Filter the Tables.jl compitable `tbl` by the logic embedded in the node

* tbl - A Tables.jl table
* node - A JLBoostTree
"""
function filter_tbl_by_splits(tbl, node::AbstractJLBoostTree)
    @assert Tables.istable(tbl)

    tblc = Tables.columns(tbl)

    # a boolean on whether to keep the row
    keeprow = keeprow_vec(tbl, node)

    view(tblc, keeprow, :)
end