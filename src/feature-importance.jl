export feature_importance

using DataFrames: DataFrame

"""
    feature_importance(jlt::JLBoostTree, df, loss, target)
    feature_importance(jlt::JLBoostTreeModel, df)
    feature_importance(jlt::JLBoostTreeModel, X, y::AbstractVector)

Return the feature of the tree computed on df
"""
feature_importance(jlt::JLBoostTreeModel, X, y::AbstractVector) = begin
    feature_importance(jlt, hcat(X, DataFrame(jlt.target => y)))
end

feature_importance(jlt::JLBoostTreeModel, df) = begin
    if (typeof(df[!, jlt.target]) <: CategoricalVector) & (jlt.loss isa LogitLogLoss)
        dfc = copy(df)
        dfc[!, jlt.target] = 2 .- categorical(dfc[!, jlt.target]).refs
        return feature_importance(trees(jlt), dfc, jlt.loss, jlt.target)
    end
    feature_importance(trees(jlt), df, jlt.loss, jlt.target)
end

add_dict!(d1, d2) = begin
    for (key, val) in d2
        if haskey(d1, key)
            d1[key] += val
        else
            d1[key] = val
        end
    end
    d1
end

dict_to_df(d) = begin
    sv = sum(values(d.freq_dict))
    d2 = Dict(key => (val/sv) for (key, val) in d.freq_dict)
    dfreq = DataFrame(feature = keys(d2) |> collect, Frequency = values(d2) |> collect)

    sv3 = sum(values(d.gain_dict))
    d3 = Dict(key => (val/sv3) for (key, val) in d.gain_dict)
    dgain = DataFrame(feature = keys(d3) |> collect, Quality_Gain = values(d3) |> collect)

    sv4 = sum(values(d.coverage_dict))
    d4 = Dict(key => (val/sv4) for (key, val) in d.coverage_dict)
    dcoverage = DataFrame(feature = keys(d4) |> collect, Coverage = values(d4) |> collect)

    sort!(join(join(dgain, dcoverage, on = :feature),  dfreq, on = :feature), [:Quality_Gain, :Coverage, :Frequency], rev = true)
end

feature_importance(jlt::AbstractVector{<:AbstractJLBoostTree}, df, loss, target) = begin
    res = feature_importance!.(jlt, Ref(df), Ref(loss), Ref(target))

    # combine
    freq_dict = res[1].freq_dict
    gain_dict = res[1].gain_dict
    coverage_dict = res[1].coverage_dict

    for r in @view(res[2:end])
        add_dict!(freq_dict, r.freq_dict)
        add_dict!(gain_dict, r.gain_dict)
        add_dict!(coverage_dict, r.coverage_dict)
    end
    d = (freq_dict = freq_dict, gain_dict = gain_dict, coverage_dict = coverage_dict)

    dict_to_df(d)
end


feature_importance(jlt::AbstractJLBoostTree, df, loss, target) = begin
    d = feature_importance!(jlt, df, loss, target)
    dict_to_df(d)
end


feature_importance!(jlt::AbstractJLBoostTree, df, loss, target, rows_bool = fill(true, nrow(df)), freq_dict = Dict{Symbol, Int}(), gain_dict = Dict{Symbol, Float64}(), coverage_dict = Dict{Symbol, Float64}(), Gs = JLBoost.g.(loss, getproperty(Tables.columns(df), target), jlt.weight), Hs = JLBoost.h.(loss, getproperty(Tables.columns(df), target), jlt.weight)) = begin
    if !isequal(jlt.splitfeature, missing)
        # compute the Quality/Gain. Coverage
        rows_bool_left = rows_bool .& (getproperty(Tables.columns(df), jlt.splitfeature) .<= jlt.split)

        rows_bool_right = rows_bool .& (.!rows_bool_left)

        G_left = sum(@view(Gs[rows_bool_left]))
        H_left = sum(@view(Hs[rows_bool_left]))
        G_right = sum(@view(Gs[rows_bool_right]))
        H_right = sum(@view(Hs[rows_bool_right]))

        # note that hyper parameters are not used to compute the gain
        #gain = G_left^2/H_left + G_right^2/H_right - (G_left + G_right)^2/(H_left + H_right)
        gain = (H_left == 0 ? 0 : G_left^2/H_left) + (H_right == 0 ? 0 : G_right^2/H_right) - (G_left + G_right)^2/(H_left + H_right)
        coverage = H_left + H_right

        if haskey(freq_dict, jlt.splitfeature)
            freq_dict[jlt.splitfeature] += 1
            gain_dict[jlt.splitfeature] += gain
            coverage_dict[jlt.splitfeature] += coverage
        else
            freq_dict[jlt.splitfeature] = 1
            gain_dict[jlt.splitfeature] = gain
            coverage_dict[jlt.splitfeature] = coverage
        end

        feature_importance!(jlt.children[1], df, loss, target, rows_bool_left,  freq_dict, gain_dict, coverage_dict, Gs, Hs)
        feature_importance!(jlt.children[2], df, loss, target, rows_bool_right, freq_dict, gain_dict, coverage_dict, Gs, Hs)
    end

    (freq_dict = freq_dict, gain_dict = gain_dict, coverage_dict = coverage_dict)
end

if false
    feature_importance(tree[1], gmsc, LogitLogLoss(), :SeriousDlqin2yrs)
end
