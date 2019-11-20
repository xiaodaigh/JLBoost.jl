export jlboost!, jlboost

"""
	jlboost(df, target, features = setdiff(names(df), (target, prev_w, new_weight)); nrounds = 1, eta = 0.3, lambda = 0, gamma = 0, max_depth = 6, subsample = 1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, verbose = false)

Fit a tree boosting model with a DataFrame, df, and target symbol and allowed features.

This is based on the xgboost interface, where possible the parameters have the same name as xgboost, see
https://xgboost.readthedocs.io/en/latest/parameter.html

* nrounds: Number of trees to fit
* base_score: global bias
* eta: The learning rate. Also the weight of each tree in the final summation of trees
* lambda: XGBoost lambda hyperparameter
* gamma: XGBoost gamma hyperparameter
* max_depth: the maximum depth of each tree
* subsample: 0-1, the proportion of rows to subsample for each tree build
* verbose: Print more information
* min_child_weight:
* colsample_bytree: Not yet implemented
* colsample_bylevel: Not yet implemented
* colsample_bynode: Not yet implemented
"""
jlboost(df, target::Symbol, warm_start; kwargs...) = jlboost(df, target, setdiff(names(df), (target,)), df->predict(object(warm_start), df))

jlboost(df, target::Symbol, features::AbstractVector{Symbol}, warm_start;	kwargs...) = jlboost(df, target, features, df->predict(warm_start, df))


function jlboost(df, target::Symbol, features::AbstractVector{Symbol} = setdiff(names(df), (target,)), warm_start::Function = df->predict(JLBoostTree(0.0), df);
	nrounds = 1, subsample = 1, eta = 1, verbose =false, kwargs...)
	# eta = 1, lambda = 0, gamma = 0, max_depth = 6,  min_child_weight = 1,
	# colsample_bytree = 1, colsample_bylevel = 1,  colsample_bynode = 1,

	@assert nrounds >= 1
	@assert subsample <= 1 && subsample > 0

	@warn "only binary classification is supported"
	objective = LogitProbLoss()

	if eta != 1
		@warn "eta != 1 is not implemented yet"
	end

	res_jlt = Vector{JLBoostTree{Float64}}(undef, nrounds)

	if subsample == 1
		new_jlt = fit_tree(objective, df, target, features, JLBoostTree(0.0), warm_start; verbose = verbose, kwargs...)
	else
		rows = sample(collect(1:nrow(df)), Int(round(nrow(df)*subsample)); replace = false)
		new_jlt = fit_tree(objective, @view(df[rows, :]), target, features, JLBoostTree(0.0), warm_start; verbose = verbose, kwargs...)
	end
	res_jlt[1] = deepcopy(new_jlt)
	push!(warm_start, res_jlt[1])

	for nround in 2:nrounds
		if verbose
			println("Fitting tree #$(nround)")
			println("creating new column weight$(nround-1) to store weight of previous tree")
		end
		# assign the previous weight

		if subsample == 1
			new_jlt = fit_tree(objective, df, target, features, JLBoostTree(0.0), warm_start; verbose = verbose, kwargs...)
		else
			rows = sample(collect(1:nrow(df)), Int(round(nrow(df)*subsample)); replace = false)
			new_jlt = fit_tree(objective, @view(df[rows, :]), target, features, JLBoostTree(0.0), warm_start; verbose = verbose, kwargs...)
		end
	    res_jlt[nround] = deepcopy(new_jlt)
	    push!(warm_start, res_jlt[nround])
	end
	res_jlt
end
