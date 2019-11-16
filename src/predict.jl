export predict

function predict(jlt::JLBoostTreeNode, df)
	# TODO a more efficient algorithm. Currently there are too many assignbools being	
	# stores the results
	res = Vector{Float64}(undef, nrow(df))
	res .= 0.0

	# stores the assignment array
	assignbool = trues(nrow(df))

    predict!(jlt, df, res, assignbool)
end

function predict(jlts::AbstractVector{JLBoostTreeNode{T}}, df) where T
	mapreduce(x->predict(x, df), +, jlts)
end

function predict!(jlt::JLBoostTreeNode, df, res, assignbool)
	if length(jlt.children) == 2
	    new_assignbool = assignbool .& (df[!, jlt.splitfeature] .<= jlt.split)
	    predict!(jlt.children[1], df, res, new_assignbool)

	    new_assignbool .= assignbool .& (df[!, jlt.splitfeature] .> jlt.split)
	    predict!(jlt.children[2], df, res, new_assignbool)
	else length(jlt.children) == 0
	    res[assignbool] .= res[assignbool] .+ jlt.weight
	end
	res
end