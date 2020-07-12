using StatsBase: sample

function select_col_sampling_strategy(colsample)
     # a sample of the features
    if 0 < colsample < 1
        col_sampling_bytree_strategy =
            function (features, args...; kwargs...)
                sample(features, floor(Int, length(features)*colsample), replace=false)
            end
    elseif colsample == 1
        col_sampling_bytree_strategy = (features, args...; kwargs...) -> features
    else
        error("colsample must be within [0, 1)")
    end

    col_sampling_bytree_strategy
end

