export AUC, gini, CrossEntropy

using DataFrames: by, DataFrame, sort!
using CategoricalArrays: CategoricalVector

import MLJBase: CrossEntropy

CrossEntropy(x, y::CategoricalVector) = CrossEntropy(x, y.refs .- 1)


function AUC_plot_data(score, target::CategoricalVector;  kwargs...)
    @assert length(levels(target)) == 2
    _AUC_plot_data(score, 2 .- target.refs; kwargs...)
end

AUC_plot_data(score, target;  kwargs...) = _AUC_plot_data(score, target;  kwargs...)

function _AUC_plot_data(score, target;  plotauc = false)
    tmpdf = by(
        DataFrame(score = score, target = target),
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

    cu, cutarget
end

"""
    AUC(score, target; plotauc = false)

Return the AUC. To generate a plot set `plotauc=true`.
"""
function AUC(score, target; kwargs...)
    cu, cutarget = AUC_plot_data(score, target; kwargs...)
    sum((cutarget[2:end] .+ cutarget[1:end-1]).*(cu[2:end].-cu[1:end-1])./2)
end

"""
    gini(score, target; plotauc = false)

Return the `gini = (AUC - 0.5)/0.05 = 2AUC - 1`. AUC is the area under the curve while gini
is the ratio of (AUC minus the area of the bottom triangle) vs (Area of upper triangle).

For AUC a random model has AUC = 0.5 but for gini a random model has gini = 0.0

To generate a plot set `plotauc=true`.
"""
function gini(score, target; plotauc = false)
    auc = AUC(score,target; plotauc = plotauc)
    2*auc-1
end
