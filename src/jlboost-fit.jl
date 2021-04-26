export jlboost!, jlboost

using DataFrames: nrow, ncol
using Tables

using ..JLBoostTrees: JLBoostTree

"""
    jlboost(df, target, features = setdiff(names(df), (target, prev_w, new_weight)),
        warm_start = fill(0.0, nrow(df)); nrounds = 1, eta = 0.3, lambda = 0, gamma = 0,
        max_depth = 6, subsample = 1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1,
        verbose = false)

Fit a tree boosting model with a DataFrame, df, and target symbol and allowed features.

This is based on the xgboost interface, where possible the parameters have the same name as xgboost,
see https://xgboost.readthedocs.io/en/latest/parameter.html

* nrounds: Number of trees to fit
* warmstart: A vector of weights from which to start training. Defaults to 0. The warmstart may be
    different for every row. This is designed to allow the model to improve upon existing models.
* eta: The learning rate. Also known as the weight of each tree in the final summation of trees
* lambda: XGBoost lambda hyperparameter
* gamma: XGBoost gamma hyperparameter
* max_depth: the maximum depth of each tree
* subsample: 0-1, the proportion of rows to subsample for each tree build
* verbose: Print more information
* colsample_bytree: (0-1] The proportion of feature column to sample for each tree.
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
    warm_start = fill(0.0, nrow(df))
	jlboost(df, target, setdiff(Tables.columnnames(df), [target]), warm_start; kwargs...)
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

function jlboost(df, target::Union{Symbol, String}, features::AbstractVector,
    warm_start::AbstractVector, loss = LogitLogLoss();
    subsample = 1, colsample_bytree = 1, max_depth = 6, max_leaves = 0, kwargs...)

    @assert 0 < subsample <= 1
    @assert 0 < colsample_bytree <= 1
    @assert Tables.istable(df)

    target = Symbol(target)
    features = Symbol.(features)

    # a sample of the rows
    row_sampling_bytree_strategy = select_row_sampling_strategy(subsample)

    # a function to sample the columns
    col_sampling_bytree_strategy = select_col_sampling_strategy(colsample_bytree)

    if max_leaves > 0
        if max_depth > 0
            @warn "You have set max_leaves=$max_leaves but max_depth > 0. The max_depth parameter is ignored."
        end
        tree_growth = lossguide
        stopping_criterion = max_leaves_stopping_criterion(max_leaves)
    else
        tree_growth = depth_wise
        stopping_criterion = max_depth_stopping_criterion(max_depth)
    end

    # TODO look at target column and provide a possible selection of loss
    # e.g. if the target is numeric then RSMELoss is more appropriate
    jlboost(df, target, features, warm_start, loss,
            row_sampling_bytree_strategy,
            col_sampling_bytree_strategy,
            tree_growth,
            stopping_criterion; kwargs...)
end

# the most canonical version of jlboost is here
function jlboost(df, target, features, warm_start::AbstractVector,
    loss,
    row_sampling_strategy::Function,
    col_sampling_bytree_strategy::Function,
    tree_growth::Function,
    stopping_criterion::Function;
	nrounds = 1, eta = 1.0, verbose = false, kwargs...)
    # eta = 1, lambda = 0, gamma = 0,  min_child_weight = 1, colsample_bylevel = 1, colsample_bynode = 1,
	#, ,  colsample_bynode = 1,

    @assert nrounds >= 1
	@assert Tables.istable(df)



    target = Symbol(target)
    features = Symbol.(features)

	dfc = Tables.columns(df)

    # res_jlt = result JLBoost trees
	res_jlt = AbstractJLBoostTree[]

    # fit the next round
	for nround in 1:nrounds
		if verbose
			println("Fitting tree #$(nround)")
		end

        # sample new columns
		features_sample = col_sampling_bytree_strategy(features, df, target, warm_start, loss;
                                                   nrounds=nrounds, eta=eta, kwargs...)
        # dfs = DataFrame Sampled
        dfs = row_sampling_strategy(dfc)
        if nround == 1
            warm_start = fill(0.0, nrow(dfs))
        else
            warm_start = predict(res_jlt[1:nround-1], dfs)
        end

        println(nround)
        println(dfc)

        new_jlt = _fit_tree!(loss, dfc, target, features_sample, warm_start, JLBoostTree(0.0),
                             tree_growth,
                             stopping_criterion; verbose=verbose, kwargs...);

        println("mehmehmehmeh")
        # added a new round of tree
        push!(res_jlt, eta*deepcopy(new_jlt))
	end
	res_jlt


    JLBoostTreeModel(res_jlt, loss, target)
end