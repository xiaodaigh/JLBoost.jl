export AUC, gini

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
