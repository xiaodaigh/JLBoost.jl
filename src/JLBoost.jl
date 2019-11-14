module JLBoost

using DataFrames
using SortingLab
#using StatsBase
using Zygote: gradient, hessian
# using ForwardDiff: gradient, hessian
using Base.Iterators: drop
#using RCall

export jlboost, best_split, _best_split, predict, fit_tree, logloss, jlboost!
export update_weight

include("JLBoostTree.jl")
include("diagnostics.jl")

using ..JLBoostTrees: JLBoostTreeNode

# using CuArrays
# using Flux: logitbinarycrossentropy

# set up loss functions

# alternate definition
softmax(w) = 1/(1 + exp(-w))
logloss(w, y) = -(y*log(softmax(w)) + (1-y)*log(1-softmax(w)))

# The Flux implemnetation
# logloss = logitbinarycrossentropy

g(loss_fn, y, prev_w) = begin
    gres = gradient(x->loss_fn(x, y), prev_w)
    gres[1]
end

h(loss_fn, y, prev_w) = begin
    hres = hessian(x->loss_fn(x[1], y), [prev_w])
    hres[1]
end

# g_forwarddiff(loss_fn, y, prev_w) = begin
#     gres = ForwardDiff.gradient(x->loss_fn(x[1], y), [prev_w])
#     gres[1]
# end

# h_forwarddiff(loss_fn, y, prev_w) = begin
#     hres = ForwardDiff.hessian(x->loss_fn(x[1], y), [prev_w])
#     hres[1]
# end

# update the weight once so that it starts at a better point
function update_weight(loss_fn, df, target, prev_w, lambda)
    target_vec = df[!, target];
    prev_w_vec = df[!, prev_w];

    -sum(g.(loss_fn, target_vec, prev_w_vec))/(sum(h.(loss_fn, target_vec, prev_w_vec)) + lambda)
end

"""
	apply_split(df::AbstractDataFrame, feature, split_at, lweight, rweight)

Apply split to the dataframe df
"""
function apply_split(df::AbstractDataFrame, feature, split_at, lweight, rweight)
    df[df[feature] .<= split_at,:prev_w] = df[df[feature] .<= split_at,:prev_w] .+ lweight
    df[df[feature] .> split_at,:prev_w] = df[df[feature] .> split_at,:prev_w] .+ rweight
    df
end

"""
	best_split(loss_fn, df::AbstractDataFrame, feature, target, prev_w, lambda, gamma; verbose = false)

Determine the best split of a given variable
"""
function best_split(loss_fn, df::DataFrame, feature, target, prev_w, lambda, gamma; verbose = false)
    if verbose
        println("Choosing a split on", feature)
    end
    df2 = sort(df[!, [target, feature, prev_w]], feature)

    x = df2[!, feature];
    target_vec = df2[!, target];
    prev_w_vec = df2[!, prev_w];

    split_res = best_split(loss_fn, x, target_vec, prev_w_vec, lambda, gamma, verbose)
    (feature = feature, split_res...)
end



"""
    best_split(loss_fn, feature, target, prev_w, lambda, gamma)

Find the best (binary) split point by loss_fn(feature, target) given a sorted iterator
of feature
"""
function best_split(loss_fn, feature, target, prev_w, lambda::Number, gamma::Number, verbose = false)
	@assert length(feature) == length(target)
	@assert length(feature) == length(prev_w)
    if issorted(feature)
        res = _best_split(loss_fn, feature, target, prev_w, lambda, gamma, verbose)
    else
        s = fsortperm(feature)
        res = _best_split(loss_fn, @view(feature[s]), @view(target[s]), @view(prev_w[s]), lambda, gamma, verbose)
    end    
end

function _best_split_old(loss_fn, feature::AbstractVector, target::AbstractVector, prev_w::AbstractVector, lambda::Number, gamma::Number, verbose = false)
    # TODO redo this using iterators
    cg = cumsum(g.(loss_fn, target, prev_w))
    ch = cumsum(h.(loss_fn, target, prev_w))

    max_cg = cg[end]
    max_ch = ch[end]

    left_split = (cg).^(2) ./(ch .+ lambda)
    right_split = (max_cg.-cg).^(2) ./ ((max_ch .- ch) .+ lambda)
    no_split = max_cg^2 /(max_ch + lambda)

    # this is the gain if we choose the cut points there
    lrn = left_split .+  right_split .- no_split .- gamma

    # TODO there could be dups in feature

    # cutting at the last point is the same as having the whole node and not split
    cutpt = findmax(@view(lrn[2:end]))[2]

    lweight = -cg[cutpt]/(ch[cutpt]+lambda)
    rweight = -(max_cg - cg[cutpt])/(max_ch - ch[cutpt] + lambda)

    (split_at = feature[cutpt], gain = lrn[cutpt], lweight = lweight, rweight = rweight)
end

"""
	_best_split(fn, f, t, p, lambda, gamma, verbose)

Assume that f, t, p are iterable
"""
function _best_split(loss_fn, feature, target, prev_w, lambda::Number, gamma::Number, verbose = false)
	cg = cumsum(g.(loss_fn, target, prev_w))
    ch = cumsum(h.(loss_fn, target, prev_w))

    max_cg = cg[end]
    max_ch = ch[end]    

    last_feature = feature[1]
    cutpt::Int = zero(Int)
    lweight::Float64 = 0.0
    rweight::Float64 = 0.0
    best_gain::Float64 = typemin(Float64)

    if length(feature) == 1
    	no_split = max_cg^2 /(max_ch + lambda)
    	gain = no_split - gamma
    	cutpt = 0
    	# lweight = -cg[cutpt]/(ch[cutpt]+lambda)
    	# rweight = -(max_cg - cg[cutpt])/(max_ch - ch[cutpt] + lambda)
    	lweight = typemin(eltype(feature))
    	rweight = typemin(eltype(feature))
	else
		for (i, (f, cg, ch)) in enumerate(zip(drop(feature,1) , @view(cg[1:end-1]), @view(ch[1:end-1])))
			if f != last_feature
				left_split = cg^2 /(ch + lambda)
				right_split = (max_cg-cg)^(2) / ((max_ch - ch) + lambda)
				no_split = max_cg^2 /(max_ch + lambda)
				gain = left_split +  right_split - no_split - gamma
				if gain > best_gain				
					best_gain = gain
					cutpt = i
					lweight = -cg/(ch+lambda)
					rweight = -(max_cg - cg)/(max_ch - ch + lambda)
				end
				last_feature = f    		
			end
		end		
	end
    
    split_at = typemin(eltype(feature))
    if cutpt >= 1
    	split_at = feature[cutpt]
    end

    (split_at = split_at, cutpt = cutpt, gain = best_gain, lweight = lweight, rweight = rweight)
end

# The main XGBoost function
function fit_tree!(df::AbstractDataFrame, target::Symbol, features::AbstractVector{Symbol}; kwargs...)
	# TODO keep only do not bend
    jlt = JLBoostTreeNode(0.0)
    fit_tree!(df, target, features, jlt; kwargs...)
end


function fit_tree!(df::AbstractDataFrame, target, features, jlt::JLBoostTreeNode;  prev_w = :prev_w, eta = 0.3, lambda = 0, gamma = 0, maxdepth = 6, verbose = false)
	# initialise the weights to 0 if the column doesn't exist yet
	if !(prev_w  in names(df))
		println("not found lah $prev_w")
	    df[!, prev_w] .= 0.0
	end	

	# compute the gain for all splits for all features
	all_splits = [best_split(logloss, df, feature, target, prev_w, lambda, gamma) for feature in features]    
	split_with_best_gain = all_splits[findmax(map(x->x.gain, all_splits))[2]]

	# there needs to be positive gain then apply split to the tree
	if split_with_best_gain.gain > 0
	    # set the parent tree node
	    jlt.split = split_with_best_gain.split_at
	    jlt.splitfeature = split_with_best_gain.feature      

	    left_treenode = JLBoostTreeNode(split_with_best_gain.lweight)        
	    right_treenode = JLBoostTreeNode(split_with_best_gain.rweight)

	    if maxdepth > 1
	        # now recursively apply the weights to left branch and right branch
	        df_left = df[df[!, split_with_best_gain.feature] .<= split_with_best_gain.split_at,:]
	        df_right = df[df[!, split_with_best_gain.feature] .> split_with_best_gain.split_at,:]

	        left_treenode  = fit_tree!(df_left,  target, features, left_treenode;  prev_w = prev_w, eta = eta, lambda = lambda, gamma = gamma, maxdepth = maxdepth - 1)
	        right_treenode = fit_tree!(df_right, target, features, right_treenode; prev_w = prev_w, eta = eta, lambda = lambda, gamma = gamma, maxdepth = maxdepth - 1)
	    end
	    jlt.children = [left_treenode, right_treenode]
	end
	jlt
end

"""
	jlboost!(df, target, features; nrounds = 1, prev_w = :prev_w, verbose = false, eta = 0.3, lambda = 0, gamma = 0, maxdepth = 6, subsample = 1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1)

Fit a tree boosting model with a DataFrame, df, and target symbol and allowed features. 

This is based on the xgboost interface, where possible the parameters have the same name as xgboost, see 
https://xgboost.readthedocs.io/en/latest/parameter.html


* nrounds: Number of trees to fit
* base_score
* eta: The learning rate. Also the weight of each tree in the final summation of trees
* lambda: XGBoost lambda hyperparameter
* gamma: XGBoost gamma hyperparameter
* maxdepth: the maximum depth of each tree
* subsample: 0-1, the proportion of rows to subsample for each tree build
* prev_w: A Symbol indicating the column containing the previous fitted weights for a warm start
* verbose: Print more information
* min_child_weight: 
* colsample_bytree: Not yet implemented
* colsample_bylevel: Not yet implemented
* colsample_bynode: Not yet implemented
"""
function jlboost!(df, target, features; kwargs...)
	jlt = JLBoostTreeNode(0.0)
	jlboost!(df, target, features, jlt; kwargs...)
end

function jlboost!(df, target, features, jlt::JLBoostTreeNode; 
	nrounds = 1, prev_w = :prev_w, base_score = 0.0, subsample = 1, 
	colsample_bytree = 1, colsample_bylevel = 1, colsample_bynode = 1, verbose = false, kwargs...)

	res_jlt = Vector{JLBoostTreeNode}(undef, nrounds)

	new_jlt = fit_tree!(df, target, features, jlt; kwargs...)
	res_jlt[1] = deepcopy(new_jlt)

	for nround in 2:nrounds
		if verbose
			println("Fitting tree #$(nround)")
			println("creating new column weight$(nround-1) to store weight of previous tree")
		end
		# assign the previous weight
		df[!, Symbol("weight"*string(nround-1))] = predict(new_jlt, df)				
		new_jlt = fit_tree!(df, target, features, new_jlt; prev_w = Symbol("weight"*string(nround-1)), kwargs...)
	    res_jlt[nround] = deepcopy(new_jlt)
	end
	res_jlt
end

function predict(jlt, df, base_score = 0.5)
	# TODO a more efficient algorithm. Currently there are too many assignbools being	
	# stores the results
	res = Vector{Float64}(undef, nrow(df))
	res .= base_score

	# stores the assignment array
	assignbool = trues(nrow(df))

    predict!(jlt, df, res, assignbool)
end

function predict(jlts::AbstractVector{JLBoostTreeNode}, df, base_score = 0.5)
	mapreduce(x->predict(x, df, base_score), +, jlts)
end

function predict!(jlt, df, res, assignbool)
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

end # module
