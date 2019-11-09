module JLBoost

using DataFrames,StatsBase
using Zygote:gradient, hessian
#using RCall

export JLBoostTreeNode, JLBoostTree, showlah
export xgboost

include("JLBoostTree.jl")

t2one(x) = x ? 1 : 0
# set up loss functions
softmax(w) = 1/(1 + exp(-w))
logloss(w, y) = -(y*log(softmax(w)) + (1-y)*log(1-softmax(w)))

#
g(loss_fn, y, prev_w) = begin
    gres = gradient(x->loss_fn(x, y), prev_w)
    gres[1]
end

h(loss_fn, y, prev_w) = begin
    hres = hessian(x->loss_fn(x[1], y), [prev_w])
    hres[1]
end

# update the weight once so that it starts at a better point
function update_weight(loss_fn, df, target, prev_w, lambda)
    target_vec = df[target];
    prev_w_vec = df[prev_w];

    -sum(g.(loss_fn, target_vec, prev_w_vec))/(sum(h.(loss_fn, target_vec, prev_w_vec)) + lambda)
end

function apply_split(df, feature, bsplit, lweight, rweight)
    df[df[feature] .<= bsplit,:prev_w] = df[df[feature] .<= bsplit,:prev_w] .+ lweight
    df[df[feature] .> bsplit,:prev_w] = df[df[feature] .> bsplit,:prev_w] .+ rweight
    df
end
# w = update_weight(logloss, df, target, prev_w, lambda)
# df[prev_w] .+= w
# (w, unique(df[prev_w])..., softmax(w))

function best_split(loss_fn, df::DataFrame, feature, target, prev_w, lambda, gamma)
    println(feature)
    df2 = sort(df[!, [target, feature, prev_w]], feature)

    x = df2[!, feature];
    target_vec = df2[!, target];
    prev_w_vec = df2[!, prev_w];

    cg = cumsum(g.(loss_fn, target_vec, prev_w_vec))
    ch = cumsum(h.(loss_fn, target_vec, prev_w_vec))

    max_cg = cg[end]
    max_ch = ch[end]

    left_split = (cg).^(2) ./(ch .+ lambda)
    right_split = (max_cg.-cg).^(2) ./ ((max_ch .- ch) .+ lambda)
    no_split = max_cg^2 /(max_ch + lambda)
    lrn = left_split .+  right_split .- no_split .- gamma

    df2[!,:lrn] = lrn
    df2[!,:rn] = 1:size(df)[1]

    df_summ = df2[by(df2, feature, rows_to_keep = :rn => maximum).rows_to_keep, :]
    maxloc = findmax(df_summ[!,:lrn])

    # (x[maxloc[2]], maxloc)
    # df2[!,:ok] = x .<= df_summ[maxloc[2],feature]
    # by(df2, :ok, df1 -> (sum(df1[target]), size(df1)[1]))

    # store the best split for this val
    cutpt = df_summ[maxloc[2],:rn]
    lweight = -cg[cutpt]/(ch[cutpt]+lambda)
    rweight = -(max_cg - cg[cutpt])/(max_ch - ch[cutpt] + lambda)

    (feature = feature, best_split = df_summ[maxloc[2],feature], gain = maxloc[1], lweight=lweight, rweight=rweight)
end

# The main XGBoost function
function xgboost(df, target, features; prev_w = :prev_w, eta = 0.3, lambda = 0, gamma = 0, maxdepth = 6, subsample = 1)
    #jlt = JLBoostTrees.JLBoostTree(JLBoostTrees.JLBoostTreeNode(0.0), df, target, features, prev_w, eta, lambda, gamma, maxdepth, subsample)
    jlt = JLBoostTrees.JLBoostTreeNode(0.0)
    xgboost(df, target, features, jlt, prev_w = prev_w, eta = eta, lambda =lambda, gamma = gamma, maxdepth = maxdepth, subsample = subsample)
end

function xgboost(df, target, features, jlt::JLBoostTrees.JLBoostTreeNode; prev_w = :prev_w, eta = 0.3, lambda = 0, gamma = 0, maxdepth = 6, subsample = 1)
    #println(maxdepth)

    # initialise the weights to 0 if the column doesn't exist yet
    if all(prev_w  .!= names(df))
        df[!, prev_w] = 0.0
    end

    # add the weight of the parent node to the weights
    # if this the first tree being fitted it is likley to be 0
    #df[prev_w] = df[prev_w] .+ jlt.weight

    # compute the gain for all splits for all features
    all_splits = [best_split(logloss, df, feature, target, prev_w, lambda, gamma) for feature in features]
    split_with_best_gain = all_splits[findmax(sortperm(all_splits, by = x -> x.gain))[2]]

    # there needs to be positive gain then apply split to the tree
    if split_with_best_gain.gain > 0
        # set the parent tree node
        jlt.split = split_with_best_gain.best_split
        jlt.splitfeature = split_with_best_gain.feature

        left_treenode = JLBoostTrees.JLBoostTreeNode(split_with_best_gain.lweight)
        right_treenode = JLBoostTrees.JLBoostTreeNode(split_with_best_gain.rweight)

        if maxdepth > 1
            # now recursively apply the weights to left branch and right branch
            df_left = df[df[split_with_best_gain.feature] .<= split_with_best_gain.best_split,:]
            df_right = df[df[split_with_best_gain.feature] .> split_with_best_gain.best_split,:]

            left_treenode = xgboost(df_left, target, features, left_treenode; prev_w = prev_w, eta = eta, lambda =lambda, gamma = gamma, maxdepth = maxdepth - 1, subsample = subsample)
            right_treenode = xgboost(df_right, target, features, right_treenode; prev_w = prev_w, eta = eta, lambda =lambda, gamma = gamma, maxdepth = maxdepth - 1, subsample = subsample)
        end
        jlt.children = [left_treenode, right_treenode]
    end
    jlt
end

# what's the best way to show the information
function scoretree(df, jlt, weight_sym)
    assignbool = trues(size(df)[1])
    if all(weight_sym .!= names(df))
        df[weight_sym] = 0.0
    end
    _scoretree!(df, jlt, weight_sym, assignbool)
end

function _scoretree!(df, jlt, weight_sym, assignbool)
    # add on the base weight
    #println((sum(assignbool), jlt.weight, jlt.splitfeature, jlt.split))
    if length(jlt.children) == 2
        new_assignbool = assignbool .& (df[jlt.splitfeature] .<= jlt.split)
        _scoretree!(df, jlt.children[1], weight_sym, new_assignbool)

        new_assignbool = assignbool .& (df[jlt.splitfeature] .> jlt.split)
        _scoretree!(df, jlt.children[2], weight_sym, new_assignbool)
    else length(jlt.children) == 0
        df[assignbool, weight_sym] = df[assignbool, weight_sym] .+ jlt.weight
    end

    df
end


function AUC(score, target; plotauc = false)
    tmpdf = by(
        DataFrame(score=score, target = target),
        :score,
        df1->DataFrame(ntarget = sum(df1[!,:target]), n = size(df1)[1])
    )
    sort!(tmpdf,:score)
    nrows = length(score)
    cutarget = accumulate(+, tmpdf[!,:ntarget]) ./ sum(tmpdf[!,:ntarget])
    cu = accumulate(+, tmpdf[!,:n]) ./ sum(tmpdf[!,:n])

    #tmpdf = DataFrame(score=score, target = target)

    #sort!(tmpdf,[!,:score, :target])
    #nrows = length(score)
    #cutarget = accumulate(+, tmpdf[!,:target]) ./ sum(tmpdf[!,:target])
    #cu = (1:nrows)./nrows

    if plotauc
        plot(vcat(0,cu), vcat(0,cutarget))
        plot!([0,1],[0,1])
    end
    (sum((cutarget[2:end] .+ cutarget[1:end-1]).*(cu[2:end].-cu[1:end-1])./2), (cu, cutarget))
end

function gini(score, target; plotauc = false)
    auc, data = AUC(score,target; plotauc = plotauc)
    (2*auc-1, data)
end


end # module
