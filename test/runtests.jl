using JLBoost
using Test
using DataFrames

@testset "smoke test" begin
    df = DataFrame(x=[1,1,1,0,0], y = [1,1,1,0,0])
    jlboost(df, :y; nrounds=4)
end
