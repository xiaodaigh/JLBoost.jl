using CuArrays

export best_split

# function best_split(loss_fn, df, feature::Symbol, target::Symbol, warmstart::AbstractVector, lambda, gamma; verbose = false)
#     if verbose
#         println("Choosing a split on ", feature)
#     end
#
#     df2 = df[:, [target, feature]]
#     df2[!, Symbol("  __jl_boost_zj_is_awesome__  ")] = warmstart
#
#     sort!(df2, feature)
#
#     x = df2[!, feature];
#     target_vec = df2[!, target];
#     warmstart_vec = df2[!, Symbol("  __jl_boost_zj_is_awesome__  ")];
#
#     split_res = best_split(loss_fn, x, target_vec, warmstart_vec, lambda, gamma, verbose)
#     (feature = feature, split_res...)
# end

# ZJ: disallowing best_split to operate on DataFrames as using DataFrames sort may not be fast
# """
# 	best_split(loss_fn, df::T, feature, target, warmstart, lambda, gamma; verbose = false) where T <: SupportedDFTypes
#
# Determine the best split of a given variable
# """
# function best_split(loss_fn, df, feature::Symbol, target::Symbol, warmstart::Symbol, lambda, gamma; verbose = false)
#     if verbose
#         println("Choosing a split on ", feature)
#     end
#     df2 = sort(df[!, [target, feature, warmstart]], feature)
#
#     x = df2[!, feature];
#     target_vec = df2[!, target];
#     warmstart_vec = df2[!, warmstart];
#
#     split_res = best_split(loss_fn, x, target_vec, warmstart_vec, lambda, gamma, verbose)
#     (feature = feature, split_res...)
# end


"""
    best_split(loss_fn, feature, target, warmstart, lambda, gamma)

Find the best (binary) split point by optimizing ∑ loss_fn(warmstart + δx, target) using order-2 Taylor series expexpansion.

Feature, target, and warmstart sorted.
"""
function best_split(loss_fn, feature::AbstractVector, target::AbstractVector, warmstart::AbstractVector, lambda::Number, gamma::Number; kwargs...)
	@assert length(feature) == length(target)
	@assert length(feature) == length(warmstart)
    if issorted(feature)
        res = _best_split(loss_fn, feature, target, warmstart, lambda, gamma; kwargs...)
    else
        s = fsortperm(feature)
        res = _best_split(loss_fn, @view(feature[s]), @view(target[s]), @view(warmstart[s]), lambda, gamma; kwargs...)
    end
end

"""
	_best_split(fn, f, t, p, lambda, gamma, verbose)

Assume that f, t, p are iterable and that they are sorted
"""
# TODO GPU friendly code
function _best_split(loss_fn, feature, target, warmstart, lambda::Number, gamma::Number; verbose = false)
	cg = cumsum(g.(loss_fn, target, warmstart))
    ch = cumsum(h.(loss_fn, target, warmstart))

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

function _best_split(loss_fn, feature::CuArray, target::CuArray, warmstart::CuArray, lambda::Number, gamma::Number; verbose = false)
	g1 = g.(loss_fn, target, warmstart)
	h1 = h.(loss_fn, target, warmstart)

	max_cg = sum(g1)
	max_ch = sum(ch)
	lambda = 0.0
	gamma = 0.0
	left_split = cumsum(g1).^2 ./ (ch .+ lambda)
	right_split = (max_cg .- cumsum(g1)).^(2) ./ ((max_ch .- ch) .+ lambda)
	no_split = max_cg^2 / (max_ch + lambda)
	gain = left_split .+  right_split .- no_split .- gamma

	fg = findmax(gain)

	(split_at = fg)
end
