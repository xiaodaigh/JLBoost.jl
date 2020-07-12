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
* stopping_criterion: Function
* tree_growth: Function
    A function to control where to grow the tree
* lambda = 0
    The L1 Norm regulization constnat
* gamma = 0
    The L2 Norm regulization constnat
* colsample_bynode = 1 (NOT IMPLEMENTED YET)
    What proportion of features to sample for each node
* colsample_bylevel = 1 (NOT IMPLEMENTED YET)
    What proportion of features to sample for each level
"""
function _fit_tree!(loss, tbl, target, features, warm_start,
    jlt::AbstractJLBoostTree = JLBoostTree(0.0),
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

    # keep track of the best gains at each node as we do not want to store the gain in the tree
    best_split_dict = Dict()

    while !stopping_criterion(jlt)
        # at the beginning there is only one leaf node which is the parent for all nodes
        # amongst the end nodes compute the best split and choose the best split based on the leaf
        # nodes keep only those where a split has not been decided
        leaf_nodes = filter(x->ismissing(x.splitfeature), get_leaf_nodes(jlt))

        # used to keep the best split for each leaf_node
        split_results = []

        # for all nodes eligible for splitting
        # compute the best split feature and the best split point
        # set the split point
        for leaf_node in leaf_nodes
            keeprow = keeprow_vec(tbl, leaf_node)
            tblc_filtered = view(tblc, keeprow, :)
            warm_start_filtered = view(warm_start, keeprow)

            # compute the gain for all splits for all features
            split_with_best_gain =
                find_best_split(loss, tblc_filtered, features[1], target, warm_start_filtered,
                                lambda, gamma; verbose=verbose, kwargs...)

            for feature in Iterators.drop(features, 1)
                feature_split =
                    find_best_split(loss, tblc_filtered, feature, target, warm_start_filtered,
                                    lambda, gamma; verbose=verbose, kwargs...)
                if feature_split.gain > split_with_best_gain.gain
                    split_with_best_gain = feature_split
                end
            end

            # remember the split but do not set children
            best_split_dict[leaf_node] = split_with_best_gain
            # set the parent tree node
            leaf_node.split = split_with_best_gain.split_at
            leaf_node.splitfeature = split_with_best_gain.feature
        end

        # tree_growth phase
        # select the node to grow based on growth function
        leaf_nodes = get_leaf_nodes(jlt)
        nodes_to_split::Vector{<:AbstractJLBoostTree} = tree_growth(leaf_nodes)

        for node_to_split in nodes_to_split
            # there needs to be positive gain then apply split to the tree
            split_with_best_gain = best_split_dict[node_to_split]
            if split_with_best_gain.gain > 0
                left_treenode = JLBoostTree(split_with_best_gain.lweight, node_to_split)
                right_treenode = JLBoostTree(split_with_best_gain.rweight, node_to_split)
                node_to_split.children = [left_treenode, right_treenode]
            end
        end
    end # end !stopping_criterion(jlt)
    jlt
end
