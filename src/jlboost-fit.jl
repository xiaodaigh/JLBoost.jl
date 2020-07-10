export jlboost!, jlboost

using DataFrames: nrow, ncol
using Tables

"""
	jlboost(df, target, features = setdiff(names(df), (target, prev_w, new_weight)), warm_start = fill(0.0, nrow(df)); nrounds = 1, eta = 0.3, lambda = 0, gamma = 0, max_depth = 6, subsample = 1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, verbose = false)

Fit a tree boosting model with a DataFrame, df, and target symbol and allowed features.

This is based on the xgboost interface, where possible the parameters have the same name as xgboost, see
https://xgboost.readthedocs.io/en/latest/parameter.html

* nrounds: Number of trees to fit
* warmstart: A vector of weights from which to start training. Defaults to 0. The warmstart may be
    different for every row. This is designed to allow the model to improve upon existing models.
* eta: The learning rate. Also known as the weight of each tree in the final summation of trees
* lambda: XGBoost lambda hyperparameter
* gamma: XGBoost gamma hyperparameter
* max_depth: the maximum depth of each tree
* subsample: 0-1, the proportion of rows to subsample for each tree build
* verbose: Print more information
* colsample_bytree: (0-1] The proportion of feature column to sample for each tree. This
* min_child_weight: The weight that needs to be in each child node before a split can occur. The
    weight is the hessian (2nd derivative) of the loss function, which happens to be 1 for squares
    loss.
* colsample_bylevel: Not yet implemented
* colsample_bynode: Not yet implemented
* monotone_contraints: Not yet implemented
* interaction_constraints: Not yet implemented
"""
function jlboost(df, target::Union{Symbol, String}; kwargs...)
    target = Symbol(target)
	jlboost(df, target, setdiff(Tables.columnnames(df), [target]), fill(0.0, nrow(df)); kwargs...)
end

function jlboost(df, target::Union{Symbol, String}, warm_start::AbstractVector{T}; kwargs...) where T <: Number
    target = Symbol(target)
	jlboost(df, target, setdiff(names(df), [target]), warm_start)
end

function jlboost(df, target::Union{Symbol, String}, features::AbstractVector{T}; kwargs...) where T <: Union{String, Symbol}
    target = Symbol(target)
    features = Symbol.(features)
	jlboost(df, target, features, fill(0.0, nrow(df)); kwargs...)
end

function jlboost(df, target::Union{Symbol, String}, features::AbstractVector, warm_start::AbstractVector, loss = LogitLogLoss();
	nrounds = 1, subsample = 1, eta = 1.0, verbose = false, colsample_bytree = 1, kwargs...)
	# eta = 1, lambda = 0, gamma = 0, max_depth = 6,  min_child_weight = 1, colsample_bylevel = 1, colsample_bynode = 1,
	#, ,  colsample_bynode = 1,
    target = Symbol(target)
    features = Symbol.(features)

	@assert nrounds >= 1
	@assert subsample <= 1 && subsample > 0
	@assert colsample_bytree <= 1 && colsample_bytree > 0
	@assert Tables.istable(df)

	dfc = Tables.columns(df)

    # res_jlt = result JLBoost trees
	res_jlt = Vector{AbstractJLBoostTree}(undef, nrounds)
	for i in 1:nrounds
	 	res_jlt[i] = JLBoostTree(0.0)
	end

    # a sample of the features
	if colsample_bytree < 1
		features_sample = sample(features, ceil(length(features)*colsample_bytree) |> Int; replace = true)
	else
		features_sample = features
	end

    # subsample some column
	if subsample == 1
		warm_start = fill(0.0, nrow(df))
		new_jlt = _fit_tree!(loss, df, target, features_sample, warm_start, JLBoostTree(0.0); verbose=verbose, kwargs...);
	else
		rows = sample(1:nrow(df), round(Int, nrow(df)*subsample); replace = false)
		warm_start = fill(0.0, length(rows))
		new_jlt = _fit_tree!(loss, view(dfc, rows, :), target, features_sample, warm_start, JLBoostTree(0.0); verbose=verbose, kwargs...);
	end
	res_jlt[1] = eta*deepcopy(new_jlt);

    # fit the next round
	for nround in 2:nrounds
		if verbose
			println("Fitting tree #$(nround)")
		end
		# assign the previous weight

		if colsample_bytree < 1
			features_sample = sample(features, ceil(Int, length(features)*colsample_bytree); replace = true)
		else
			features_sample = features
		end

		if subsample == 1
			warm_start = predict(res_jlt[1:nrounds-1], df)
			new_jlt = _fit_tree!(loss, df, target, features_sample, warm_start, JLBoostTree(0.0); verbose=verbose, kwargs...);
		else
			rows = sample(collect(1:nrow(df)), Int(round(nrow(df)*subsample)); replace = false)
			warm_start = predict(res_jlt[1:nrounds-1], view(dfc, rows, :))

			new_jlt = _fit_tree!(loss, view(df, rows, :), target, features_sample, warm_start, JLBoostTree(0.0); verbose=verbose,kwargs...);
		end
	    res_jlt[nround] = eta*deepcopy(new_jlt)
	end
	res_jlt

	JLBoostTreeModel(res_jlt, loss, target)
end

#_fit_tree!(loss, df, target, features, warm_start, nothing, jlt::JLBoostTree = JLBoostTree(0.0);
