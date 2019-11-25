#using CuArrays
using Statistics: mean
using Tables

export best_split

function best_split(loss, df, feature::Symbol, target::Symbol, warmstart::AbstractVector, lambda, gamma; verbose = false, kwargs...)
	 @assert Tables.istable(df)

	 if verbose
	     println("Choosing a split on ", feature)
	 end

	 dfc = Tables.columns(df)

	 x = getproperty(dfc, feature)
	 target_vec = getproperty(dfc, target);

	 split_res = best_split(loss, x, target_vec, warmstart, lambda, gamma; verbose = verbose, kwargs...)
	 (feature = feature, split_res...)
end


"""
    best_split(loss, feature, target, warmstart, lambda, gamma)

Find the best (binary) split point by optimizing ∑ loss(warmstart + δx, target) using order-2 Taylor series expexpansion.

Does not assume that Feature, target, and warmstart sorted and will sort them for you.
"""
function best_split(loss, feature::AbstractVector, target::AbstractVector, warmstart::AbstractVector, lambda::Number, gamma::Number; kwargs...)
	@assert length(feature) == length(target)
	@assert length(feature) == length(warmstart)
	
    if issorted(feature)
        res = _best_split(loss, feature, target, warmstart, lambda, gamma; kwargs...)
    else
        s = fsortperm(feature)
        res = _best_split(loss, @view(feature[s]), @view(target[s]), @view(warmstart[s]), lambda, gamma; kwargs...)
    end
end

"""
	_best_split(fn, f, t, p, lambda, gamma, verbose)

Assume that f, t, p are iterable and that they are sorted. Intended for advance users only
"""
function _best_split(loss, feature, target, warmstart, lambda::Number, gamma::Number; min_child_weight = 1, verbose = false)
	cg = cumsum(g.(loss, target, warmstart))
    ch = cumsum(h.(loss, target, warmstart))

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
    	lweight = -cg[end]/(ch[end]+lambda)
    	rweight = -cg[end]/(ch[end]+lambda)
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

	# set the split at the point at the end
    split_at = feature[end]

	# the child weight is the hessian
    if cutpt >= 1
    	split_at = feature[cutpt]
		if ch[cutpt] < min_child_weight
			# the weight is less than child weight; do not split further
			split_at = feature[end]
			cutpt = 0
			no_split = max_cg^2 /(max_ch + lambda)
	    	gain = no_split - gamma
	    	cutpt = 0
	    	lweight = -cg[end]/(ch[end]+lambda)
	    	rweight = -cg[end]/(ch[end]+lambda)
		end
    end

    (split_at = split_at, cutpt = cutpt, gain = best_gain, lweight = lweight, rweight = rweight)
end

# TODO more reseach into GPU friendliness
# function _best_split(loss, feature, target::CuArray, warmstart::CuArray, lambda::Number, gamma::Number; verbose = false)
# 	g1 = g.(loss, target, warmstart)
# 	h1 = h.(loss, target, warmstart)
#
# 	cg = cumsum(g1)
# 	ch = cumsum(h1)
#
# 	max_cg = sum(g1)
# 	max_ch = sum(h1)
#
# 	lambda = 0.0
# 	gamma = 0.0
# 	left_split = cumsum(g1).^2 ./ (cumsum(h1) .+ lambda)
# 	right_split = (max_cg .- cumsum(g1)).^(2) ./ ((max_ch .- cumsum(h1)) .+ lambda)
# 	no_split = max_cg^2 / (max_ch + lambda)
# 	gain = left_split .+  right_split .- no_split .- gamma
#
# 	# find the positions where values change because it's only meaningful to cut
# 	# at where the values change
# 	i = findall(!=(0), diff(feature))
#
# 	split_at, cutpt_i = findmax(gain[i])
# 	cutpt = i[cutpt_i]
# 	best_gain = gain[cutpt]
# 	lweight = -cg[cutpt] / (ch[cutpt] + lambda)
# 	rweight = -(max_cg - cg[cutpt])/(max_ch - ch[cutpt] + lambda)
#
# 	(split_at = split_at, cutpt = cutpt, gain = best_gain, lweight = lweight, rweight = rweight)
# end
