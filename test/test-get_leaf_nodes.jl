using JLBoost: get_leaf_nodes

@testset "get_leaf_nodes" begin
    jlt = JLBoostTree(1.0)

    x = get_leaf_nodes(jlt)
    println(typeof(x))
    @test x == [jlt]


    jlt.children = [JLBoostTree(1.0), JLBoostTree(1.0)]

    x = get_leaf_nodes(jlt)
    println(typeof(x))
    @test x == jlt.children
end
