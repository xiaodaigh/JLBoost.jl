export keeprow_vec, filter_tbl_by_splits

using DataFrames: nrow

"""
    Returns the boolean that can used to filter the table

* tbl - A Tables.jl table
* node - A JLBoostTree
"""
function keeprow_vec(tbl, node::AbstractJLBoostTree)
    @assert Tables.istable(tbl)

    tblc = Tables.columns(tbl)

    # a boolean on whether to keep the row
    keeprow = trues(nrow(tblc))

    while node !== nothing
        # now recursively apply the weights to left branch and right branch
        keeprow .&= getproperty(tblc, node.feature) .<= node.split_at
        node = node.parent
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