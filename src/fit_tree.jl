export fit_tree!, fit_tree

using Tables

"""
	_fit_tree(loss, df, target, features, warm_start, leaf_queue

Fit a tree by following a algorithm
"""
function _fit_tree!(loss, df, target, features, warm_start, feature_choice_strategy, jlt::JLBoostTree = JLBoostTree(0.0);
	colsample_bytree = 1, colsample_bynode = 1, colsample_bylevel = 1, lambda = 0, gamma = 0,
	max_depth = 6, verbose = false, kwargs...)

	@assert colsample_bytree <= 1 && colsample_bytree > 0
	@assert colsample_bynode <= 1 && colsample_bynode > 0
	@assert colsample_bylevel <= 1 && colsample_bylevel > 0
	@assert Tables.istable(df)

	# make absolutely sure that target is not part of it
	features = setdiff(features, [target])

	# compute the gain for all splits for all features
	split_with_best_gain = best_split(loss, df, features[1], target, warm_start, lambda, gamma; verbose=verbose, kwargs...)

	for feature in @view(features[2:end])
		new_feature = best_split(loss, df, feature, target, warm_start, lambda, gamma; verbose=verbose, kwargs...)
		if new_feature.gain > split_with_best_gain.gain
			split_with_best_gain = new_feature
		end
	end
	#return split_with_best_gain

	# there needs to be positive gain then apply split to the tree
	if split_with_best_gain.gain > 0
	    # set the parent tree node
		if verbose
			println(split_with_best_gain)
		end
	    jlt.split = split_with_best_gain.split_at
	    jlt.splitfeature = split_with_best_gain.feature

	    left_treenode = JLBoostTree(split_with_best_gain.lweight, jlt)
	    right_treenode = JLBoostTree(split_with_best_gain.rweight, jlt)
	    jlt.children = [left_treenode, right_treenode]

	    if max_depth > 1
			dfc = Tables.columns(df)
		 	 # now recursively apply the weights to left branch and right branch
			 left_bool = getproperty(dfc, split_with_best_gain.feature) .<= split_with_best_gain.split_at
			 # need at least two to consider a split
			 if sum(left_bool) > 1
			 	df_left = view(dfc, left_bool, :)
			 	warm_start_left = @view(warm_start[left_bool])
				# this will grow the left_treenode
			 	_fit_tree!(loss, df_left,  target, features, warm_start_left, nothing, left_treenode;  lambda = lambda, gamma = gamma, max_depth = max_depth - 1, verbose = verbose)
			end

			 right_bool = getproperty(dfc, split_with_best_gain.feature) .> split_with_best_gain.split_at
			 if sum(right_bool) > 1
		 	 	df_right = view(dfc, right_bool, :)
			 	warm_start_right = @view(warm_start[right_bool])
				# this will grow the right_treenode
		 	 	_fit_tree!(loss, df_right, target, features, warm_start_right, nothing, right_treenode; lambda = lambda, gamma = gamma, max_depth = max_depth - 1, verbose = verbose)
			end
	     end
	end
 	jlt
end
