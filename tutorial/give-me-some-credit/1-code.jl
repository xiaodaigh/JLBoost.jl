using Revise
using JLBoost: is_left_child
using AbstractTrees: isroot
using JLBoost, CSV, JDF, TidierData
using DataFrames

using TidierData
using TidierData: @clean_names, @group_by, @summarise
using Chain: @chain

data = @chain JDF.load("cs-training.jdf") begin
    DataFrame
    @select -(monthly_income, number_of_dependents)
end

names(data)

function fit_score_card(data, target, features)
    warm_start = fill(0.0, nrow(data))
    gini = 0
    feature_gini = Tuple{eltype(features), Float64}[]
    best_feature = features[1]
    feature_model = Dict()

    final_model = []

    while length(features) > 0
        for feature in features
            @info "Trying $(feature)"
            model = jlboost(data, target, [feature], warm_start; verbose=false)
            push!(feature_gini, (feature, AUC(-(model(data) + warm_start), data[!, target])))
            feature_model[feature] = model
        end

        sort!(feature_gini, by=x->x[2], rev=true)


        if feature_gini[1][2] > gini
            gini = feature_gini[1][2]
            best_feature = feature_gini[1][1]
            push!(final_model, feature_model[best_feature].jlt[1])
        else
            return final_model
        end

        warm_start .= warm_start .+ feature_model[best_feature](data)

        setdiff!(features, [best_feature])

        if length(features) == 0
            return final_model
        end

        best_feature = features[1]
        gini = 0
        feature_gini = Tuple{eltype(features), Float64}[]
    end

    final_model
end


fm = fit_score_card(data, :serious_dlqin2yrs, setdiff(names(data), [string("serious_dlqin2yrs")]));
fm = Vector{WeightedJLBoostTree{nothing}}(fm)

AUC(-predict(fm, data), data[!, :serious_dlqin2yrs])


convert_to_binning(tree::WeightedJLBoostTree) = convert_to_binning(tree.tree)

function binning_bounds(tree::JLBoostTree)
    if isroot(tree)
        return [-Inf, Inf]
    end


    bound = binning_bounds(tree.parent)


    # if ismissing(tree.split)
    #     return bound
    # end

    if is_left_child(tree)
        bound[2] = tree.parent.split
    else
        bound[1] = tree.parent.split
    end

    bound
end

sf(x) = isnothing(x.parent) ? x.splitfeature : x.parent.splitfeature


function convert_to_binning(tree::JLBoostTree)
    # compute the bounds for each node
    leaf_nodes = JLBoost.get_leaf_nodes(tree)

    x = [(sf(leaf_node), binning_bounds(leaf_node), leaf_node.weight) for leaf_node in leaf_nodes]

    sort!(x, by=x->(x[2][1], x[2][2]))

    # [println("($(x[2][1]), $(x[2][2])] -> $(x[3])") for x in x]

    x
end

bins = [convert_to_binning(fm.tree) for fm in fm]


df = mapreduce(vcat, bins) do bin
    DataFrame(feature = [bin[1] for bin in bin], binning = [bin[2] for bin in bin], weight = [bin[3] for bin in bin])
end