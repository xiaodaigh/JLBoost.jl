export fit_tree!, fit_tree

using Tables

"""
	_fit_tree(loss, df, target, features, warm_start, jlt, node_colsample_strategy)

Fit a tree by following a algorithm

Parameters:
* loss
    A SupervisedLoss from LossFunctions where `deriv` and `deriv2` are defined
* tbl
    A Tables.jl compatible table
* target
    The column name. Likely to be a `Symbol` or a `String`
* features
    The iterable of features name. The type of the elements should be Symbol or String
* warm_start
    A vector weight to serve as starting weights
* jlt
    The JLBoost tree to update
* node_colsample_strategy

* tree_growth: Function
    A function to control where to grow the tree
* colsample_bytree = 1
    What proportion of features to sample for each tree
* colsample_bynode = 1
    What proportion of features to sample for each node
* colsample_bylevel = 1
    What proportion of features to sample for each level
* lambda = 0
    The L1 Norm regulization constnat
* gamma = 0
    The L2 Norm regulization constnat
* max_depth = 6
    The maximum depth of the tree
"""
function _fit_tree!(loss, tbl, target, features, warm_start,
    jlt::AbstractJLBoostTree = JLBoostTree(0.0),
    col_sampling_bytree_strategy = (features, args...; kwargs...)->features,
    tree_growth = depth_wise,
    stopping_criterion = max_depth(6);
    lambda = 0, gamma = 0,
    verbose = false, #colsample_bynode = 1, colsample_bylevel = 1,
	kwargs...)

    @assert Tables.istable(tbl)

    tblc = Tables.columns(tbl)

	# make absolutely sure that target is not part of features
    features = setdiff(features, [target])

    if verbose
        println("`_fit_tree!`: Current state of tree $jlt")
    end

    while !stopping_criterion(jlt)
        # at the beginning there is only one leaf node which is the parent for all nodes
        # amongst the end nodes compute the best split and choose the best split based on the leaf
        # nodes
        leaf_nodes = get_leaf_nodes(jlt)

        for leaf_node in leaf_nodes
            tblc_sampled = sample(tblc, somehow...)
            warm_start_sampled = sample(warm_start)

            # compute the gain for all splits for all features
            split_with_best_gain =
                best_split(loss,tblc_sampled, features[1], target, warm_start_sampled, lambda,
                           gamma; verbose=verbose, kwargs...)

            for feature in Iterators.drop(features, 1)
                feature_split =
                    best_split(loss,tblc_sampled, feature, target, warm_start_sampled, lambda,
                               gamma; verbose=verbose, kwargs...)
                if feature_split.gain > split_with_best_gain.gain
                    split_with_best_gain = feature_split
                end
            end

            # there needs to be positive gain then apply split to the tree
            if split_with_best_gain.gain > 0
                if verbose
                    println(split_with_best_gain)
                end
                # set the parent tree node
                leaf_node.split = split_with_best_gain.split_at
                leaf_node.splitfeature = split_with_best_gain.feature

                left_treenode = JLBoostTree(split_with_best_gain.lweight, leaf_node)
                right_treenode = JLBoostTree(split_with_best_gain.rweight, leaf_node)
                leaf_node.children = [left_treenode, right_treenode]
                # if max_depth > 1
                #     leaf_nodes = get_leaf_nodes(jlt)

                #     tblc = Tables.columns(tbl)
                #     # now recursively apply the weights to left branch and right branch
                #     left_bool = getproperty(tblc, split_with_best_gain.feature) .<= split_with_best_gain.split_at
                #     # need at least two to consider a split
                #     if sum(left_bool) > 1
                #         tbl_left = view(tblc, left_bool, :)
                #         warm_start_left = @view(warm_start[left_bool])
                #         # this will grow the left_treenode
                #         _fit_tree!(loss, tbl_left,  target, features, warm_start_left, left_treenode;  lambda = lambda, gamma = gamma, max_depth = max_depth - 1, verbose = verbose)
                #     end

                #     right_bool = getproperty(tblc, split_with_best_gain.feature) .> split_with_best_gain.split_at
                #     if sum(right_bool) > 1
                #         tbl_right = view(tblc, right_bool, :)
                #         warm_start_right = @view(warm_start[right_bool])
                #         # this will grow the right_treenode
                #         _fit_tree!(loss, tbl_right, target, features, warm_start_right, right_treenode; lambda = lambda, gamma = gamma, max_depth = max_depth - 1, verbose = verbose)
                #     end
                # end
            end
            jlt
        end
    end
end
