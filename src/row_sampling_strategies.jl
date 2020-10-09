using StatsBase: sample
using DataFrames: nrow

function select_row_sampling_strategy(subsample)
    if 0 < subsample < 1
        row_sampling_bytree_strategy =
            function(df, args...; kwargs...)
                rows = sample(1:nrow(df), round(Int, nrow(df)*subsample); replace = false)
            end
    elseif subsample == 1
        row_sampling_bytree_strategy =
            function(df, args...; kwargs...)
                df
            end
    else
        error("`subsample` must be within [0, 1)")
    end

    row_sampling_bytree_strategy
end