export fit_tree

function fit_tree(objective, df::T, target, features, jlt::JLBoostTreeNode, warm_start;  
	colsample_bytree = 1, colsample_bynode=1, colsample_bylevel=1, lambda = 0, gamma = 0, 
	max_depth = 6, verbose = false) where T <: AbstractDataFrame

	@assert colsample_bytree <= 1 && colsample_bytree > 0
	@assert colsample_bynode <= 1 && colsample_bynode > 0
	@assert colsample_bylevel <= 1 && colsample_bylevel > 0

	# compute the gain for all splits for all features
	prev_w = predict(warm_start, df)

	all_splits = [best_split(objective, df, feature, target, prev_w, lambda, gamma; verbose=verbose) for feature in features]
	# return all_splits
	split_with_best_gain = all_splits[findmax(map(x->x.gain, all_splits))[2]]	


	# there needs to be positive gain then apply split to the tree
	if split_with_best_gain.gain > 0
	    # set the parent tree node
	    jlt.split = split_with_best_gain.split_at
	    jlt.splitfeature = split_with_best_gain.feature      

	    left_treenode = JLBoostTreeNode(split_with_best_gain.lweight)        
	    right_treenode = JLBoostTreeNode(split_with_best_gain.rweight)

	    if max_depth > 1
	        # now recursively apply the weights to left branch and right branch
	        df_left = @view(df[df[!, split_with_best_gain.feature] .<= split_with_best_gain.split_at,:])
	        df_right = @view(df[df[!, split_with_best_gain.feature] .> split_with_best_gain.split_at,:])

	        left_treenode  = fit_tree(objective, df_left,  target, features, left_treenode, warm_start; lambda = lambda, gamma = gamma, max_depth = max_depth - 1, verbose = verbose)
	        right_treenode = fit_tree(objective, df_right, target, features, right_treenode, warm_start; lambda = lambda, gamma = gamma, max_depth = max_depth - 1, verbose = verbose)
	    end
	    jlt.children = [left_treenode, right_treenode]
	end
	jlt
end


# Fit a tree using tree boosting algorithm
function fit_tree!(objective, df::AbstractDataFrame, target::Symbol, features::AbstractVector{Symbol}; kwargs...)
	# TODO keep only do not bend
    jlt = JLBoostTreeNode(0.0)
    fit_tree!(objective, df, target, features, jlt; kwargs...)
end

function fit_tree!(objective, df::T, target, features, jlt::JLBoostTreeNode;  prev_w = :prev_w, eta = 0.3, lambda = 0, gamma = 0, max_depth = 6, verbose = false) where T <: AbstractDataFrame
	# initialise the weights to 0 if the column doesn't exist yet
	if !(prev_w  in names(df))
		@warn "You have supplied `prev_w` but it's unpopulated. Initialising with value 0.0"
		if T <: SubDataFrame
	    	parent(df)[!, prev_w] .= 0.0
	    else
	    	df[!, prev_w] .= 0.0
	    end
	end

	# compute the gain for all splits for all features
	all_splits = [best_split(objective, df, feature, target, prev_w, lambda, gamma; verbose=verbose) for feature in features]    
	split_with_best_gain = all_splits[findmax(map(x->x.gain, all_splits))[2]]

	# there needs to be positive gain then apply split to the tree
	if split_with_best_gain.gain > 0
	    # set the parent tree node
	    jlt.split = split_with_best_gain.split_at
	    jlt.splitfeature = split_with_best_gain.feature      

	    left_treenode = JLBoostTreeNode(split_with_best_gain.lweight)        
	    right_treenode = JLBoostTreeNode(split_with_best_gain.rweight)

	    if max_depth > 1
	        # now recursively apply the weights to left branch and right branch
	        df_left = @view(df[df[!, split_with_best_gain.feature] .<= split_with_best_gain.split_at,:])
	        df_right = @view(df[df[!, split_with_best_gain.feature] .> split_with_best_gain.split_at,:])

	        left_treenode  = fit_tree!(objective, df_left,  target, features, left_treenode;  prev_w = prev_w, eta = eta, lambda = lambda, gamma = gamma, max_depth = max_depth - 1, verbose = verbose)
	        right_treenode = fit_tree!(objective, df_right, target, features, right_treenode; prev_w = prev_w, eta = eta, lambda = lambda, gamma = gamma, max_depth = max_depth - 1, verbose = verbose)
	    end
	    jlt.children = [left_treenode, right_treenode]
	end
	jlt
end