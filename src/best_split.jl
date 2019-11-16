export best_split

function best_split(loss_fn, df::AbstractDataFrame, feature::Symbol, target::Symbol, prev_w::AbstractVector, lambda, gamma; verbose = false)
    if verbose
        println("Choosing a split on ", feature)
    end

    df2 = df[:, [target, feature]]
    df2[!, Symbol("  __jl_boost_zj_is_awesome__  ")] = prev_w

    sort!(df2, feature)

    x = df2[!, feature];
    target_vec = df2[!, target];
    prev_w_vec = df2[!, Symbol("  __jl_boost_zj_is_awesome__  ")];

    split_res = best_split(loss_fn, x, target_vec, prev_w_vec, lambda, gamma, verbose)
    (feature = feature, split_res...)
end


"""
	best_split(loss_fn, df::AbstractDataFrame, feature, target, prev_w, lambda, gamma; verbose = false)

Determine the best split of a given variable
"""
function best_split(loss_fn, df::AbstractDataFrame, feature::Symbol, target::Symbol, prev_w::Symbol, lambda, gamma; verbose = false)
    if verbose
        println("Choosing a split on ", feature)
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
    	# lweight = -1.0
    	# rweight = -1.0
    	lweight = -cg[end]/(ch[end]+lambda)
    	rweight = -cg[end]/(ch[end]+lambda)
    	# lweight = typemin(eltype(feature))
    	# rweight = typemin(eltype(feature))
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
    
    split_at = feature[1]
    if cutpt >= 1
    	split_at = feature[cutpt]
    end

    (split_at = split_at, cutpt = cutpt, gain = best_gain, lweight = lweight, rweight = rweight)
end