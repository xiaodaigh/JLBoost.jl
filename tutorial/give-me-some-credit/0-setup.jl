using Revise
using CSV, JDF, JLBoost, TidierData
using DataFrames
using Chain: @chain
using Missings: missing


parseint(x) = x == "NA" ? missing : parse(Int, x)

function load_then_save(fn)
    data = @chain CSV.read("$fn.csv", DataFrame) begin
        @clean_names
        @mutate monthly_income = parseint(monthly_income)
        @mutate number_of_dependents = parseint(number_of_dependents)
    end

    JDF.save("$fn.jdf", data);
end

load_then_save("cs-test");
load_then_save("cs-training");

data.number_of_dependents