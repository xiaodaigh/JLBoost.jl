import StatsBase: predict

import DataFrames: AbstractDataFrame, DataFrame

export predict

"""
    predict(jlt, df)
    predict(jlt, df, T) # where T is Number type to control the output type
    jlt(df) # `jlt` is a JLBoostTreeModel

Apply the fitted model on data.

* jlt - A JLBoostTreeModel
* df - a Tables.jl compatible Table
"""

(jlt::AbstractJLBoostTree)(args...) = predict(jlt, args...)

(jltm::JLBoostTreeModel)(args...) = predict(jltm, args...)

(jlt::AbstractArray{JLBoostTreeModel})(args...) = predict(jlt, args...)

predict(jlt::JLBoostTreeModel, df) = predict(trees(jlt), df)

# defaults to Float64 (double) as output
predict(jlt::AbstractJLBoostTree, df) = predict(jlt, df, Float64)

function predict(jlt::AbstractJLBoostTree, df, out_eltype::Type{T})::Vector{T} where {T <: Number}
	# TODO a more efficient algorithm. Currently there are too many assignbools being
	# stores the results
    res = zeros(out_eltype, nrow(df))

    # stores the assignment array
    assignbool = trues(nrow(df))

    predict!(jlt, df, res, assignbool)
end

function predict(jlts, df)
    mapreduce(x -> predict(x, df), +, jlts)
end

function predict!(jlt::JLBoostTree, df, res, assignbool)
	if length(jlt.children) == 2
        tmp = getproperty(Tables.columns(df), jlt.splitfeature)

        # TODO
	    new_assignbool = assignbool .& ismissing.(tmp) .|| (tmp .<= jlt.split)
	    predict!(jlt.children[1], df, res, new_assignbool)

        new_assignbool .= assignbool .& .!(ismissing.(tmp) .|| (tmp .<= jlt.split))
	    predict!(jlt.children[2], df, res, new_assignbool)
    elseif length(jlt.children) == 0
	    res[assignbool] .+= jlt.weight
    else
        throw("JLBoost.jl: `predict!` does not support more than 2 children in a tree node yet")
	end
	res
end

function predict!(jlt::WeightedJLBoostTree, df, res, assignbool)
    jlt.eta .* predict!(jlt.tree, df, res, assignbool)
end
