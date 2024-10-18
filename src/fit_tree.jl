export fit_tree!, fit_tree


using ...JLBoostTrees: is_left_child
using Tables

function tree_diag_print(jlt)
    if isnothing(jlt.parent)
        parent_feature=nothing
        parent_split=nothing
        split_sign = ""
    else
        parent_feature=jlt.parent.splitfeature
        parent_split=jlt.parent.split
        split_sign = is_left_child(jlt) ? "<=" : ">"
    end

    if ismissing(jlt.splitfeature)
        "split not set yet; parent: $parent_feature$split_sign$parent_split"
    else
        if isnothing(parent_feature)
            "$(jlt.splitfeature) split at $(jlt.split); this is ROOT"
        else
            "$(jlt.splitfeature) split at $(jlt.split); parent: $parent_feature$split_sign$parent_split"
        end
    end
end

"""
	_fit_tree(loss, df, target, features, warm_start, jlt, node_colsample_strategy)

Fit a tree by following an algorithm

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
    jlt = JLBoostTree(0.0),
    tree_growth = depth_wise,
    stopping_criterion = max_depth_stopping_criterion(1);
    lambda = 0, gamma = 0,
    verbose = false, #colsample_bynode = 1, colsample_bylevel = 1,
	kwargs...)

    @assert Tables.istable(tbl)

    # tblc = Tables.columns(tbl)

    @assert nrow(tbl) >= 2 # seriously? you have so few records

	# make absolutely sure that target is not part of features
    if target in features
        @warn "{target} is in features; removing from features"
        features = setdiff(features, [target])
    end

    if verbose
        @info "`_fit_tree!`: Current state of tree $jlt"
    end

    no_more_gains_to_found = false

    # keep track of the best gains at each node as we do not want to store the gain in the tree
    # this shouldn't be reset and should be placed outside of the while loop below


    while !no_more_gains_to_found && !stopping_criterion(jlt)
        if verbose
            @info "BEST SPLIT PHASE: Tree Depth=$(treedepth(jlt))"
        end
        # at the beginning there is only one leaf node which is the parent for all nodes
        # amongst the end nodes compute the best split and choose the best split based on the leaf
        # nodes keep only those where a split has not been decided
        # println("the nodes considered for expansion are $(get_leaf_nodes(jlt) |> length)")

        # println.(get_leaf_nodes(jlt) .|> tree_diag_print)

        leaf_nodes = filter(x->ismissing(x.splitfeature) && ismissing(x.gain), get_leaf_nodes(jlt))

        if verbose
            @info "BEST SPLIT PHASE: $(length(leaf_nodes)) nodes are considered for expansion:"
            for leaf_node in leaf_nodes
                @info "BEST SPLIT PHASE: $(leaf_node |> tree_diag_print)"
            end
        end

        best_split_dict = Dict()

        # for all nodes eligible for splitting
        # compute the best split feature and the best split point
        # set the split point
        # leaf_node = leaf_nodes[1]
        for leaf_node in leaf_nodes
            if verbose
                @info "BEST SPLIT PHASE: Calculating best split for $(leaf_node |> tree_diag_print)"
            end

            if isnothing(leaf_node.parent)
                # if the node is the parent
                tbl_filtered = tbl
                warm_start_filtered = warm_start
            else
                keeprow = keeprow_vec(tbl, leaf_node)
                if sum(keeprow) <= 2
                    # no rows are kept, so move to next node
                    leaf_node.split = typemin(Float64)
                    leaf_node.splitfeature = Symbol("too few records for split")
                    leaf_node.gain = typemin(Float64)
                    if verbose
                        @info "BEST SPLIT PHASE: this branch has too few records $(leaf_node)"
                    end
                    continue
                end

                tbl_filtered = tbl[keeprow, :]
                warm_start_filtered = warm_start[keeprow]
            end

            # compute the gain for all splits for all features
            split_with_best_gain =
                find_best_split(loss, tbl_filtered, features[1], target, warm_start_filtered,
                                lambda, gamma; verbose=verbose, kwargs...)

            for feature in Iterators.drop(features, 1)
                feature_split =
                    find_best_split(loss, tbl_filtered, feature, target, warm_start_filtered,
                                    lambda, gamma; verbose=verbose, kwargs...)
                if feature_split.gain > split_with_best_gain.gain
                    split_with_best_gain = feature_split
                end
            end

            if verbose
                @info("BEST SPLIT PHASE: found a best split at $(split_with_best_gain.feature) <= $(split_with_best_gain.split_at); gain:$(split_with_best_gain.gain) further:$(split_with_best_gain.should_split_further) for $(leaf_node)")
            end

            best_split_dict[leaf_node] = split_with_best_gain

            # set the parent tree node
            leaf_node.split = split_with_best_gain.split_at
            leaf_node.splitfeature = split_with_best_gain.feature
            leaf_node.gain = split_with_best_gain.gain

            # reset the best split
            split_with_best_gain = ()
        end

        # tree_growth phase
        # select the node to grow based on growth function
        # the tree_growth function will return the list of
        # nodes_to_split = tree_growth(jlt)
        nodes_to_split::Vector{<:AbstractJLBoostTree} = tree_growth(jlt)
        if verbose
            @info "TREE GROWTH PHASE: Found $(length(nodes_to_split)) node-candidates to split"
        end

        no_more_gains_to_found = true
        for node_to_split in nodes_to_split
            if verbose
                # there needs to be positive gain then apply split to the tree
                # println(best_split_dict)
                @info "TREE GROWTH PHASE: Split at: $(node_to_split |> tree_diag_print)"
            end

            # BUG: seems to fail if the node only contains one value
            # @info "State of best_split_dict $best_split_dict"

            # sometimes the nodes to split was NOT considered by the best split phase
            if haskey(best_split_dict, node_to_split)
                split_with_best_gain = best_split_dict[node_to_split]

                if split_with_best_gain.should_split_further && (split_with_best_gain.gain > 0)
                    no_more_gains_to_found = false
                    left_treenode = JLBoostTree(split_with_best_gain.lweight; parent = node_to_split)
                    right_treenode = JLBoostTree(split_with_best_gain.rweight; parent = node_to_split)
                    node_to_split.children = [left_treenode, right_treenode]
                else
                    if verbose
                        @info "TREE GROWTH PHASE: NOT split further as no more gains to be found for above: $(node_to_split |> tree_diag_print)"
                    end
                end
            else
                # This seems fine as the node and these tree growth selection don't alway align
                #@warn "TREE GROWTH PHASE: SKIPPED SPLIT node_to_split not found in possible growth spots; potential logic/algorithm error $(node_to_split |> tree_diag_print) was not considered by the best split phase"
            end
        end
    end # end !stopping_criterion(jlt)
    jlt
end
