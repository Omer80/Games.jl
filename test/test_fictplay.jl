# ------------------------- #
# Testing fictitious play   #
# ------------------------- #

# NOTE: We include `@inferred` at least once for each function name. We do
#       that multiple times for the same function if we have particular reason
#       to believe there might be a type stability with that function.


using Distribution


@testset "Testing fictplay.jl" begin

    matching_pennies_bimatrix = Array{Float64}(undef, 2, 2, 2)
    matching_pennies_bimatrix[:, 1, 1] = [1, -1]
    matching_pennies_bimatrix[:, 1, 2] = [-1, 1]
    matching_pennies_bimatrix[:, 2, 1] = [-1, 1]
    matching_pennies_bimatrix[:, 2, 2] = [1, -1]
    g = NormalFormGame(matching_pennies_bimatrix)

    gain = 0.1
    init_actions = (1,1)

    @testset "Testing fictitious play" begin

        fp_dec = FictitiousPlay(g)
        fp_con = FictitiousPlay(g, ConstantGain(gain))

        @test @inferred(play(fp_dec, init_actions)) == ([1.0,0.0], [0.5,0.5])
        @test @inferred(play(fp_con, init_actions)) == ([1.0,0.0], [0.9,0.1])

        @test @inferred(time_series(fp_dec, 3, init_actions)) ==
              ([1.0 1.0 1.0; 0.0 0.0 0.0], [1.0 0.5 1/3; 0.0 0.5 2/3])
        @test @inferred(time_series(fp_con, 3 ,init_actions)) ==
              ([1.0 1.0 1.0; 0.0 0.0 0.0], [1.0 0.9 0.81; 0.0 0.1 0.19])
    end

    @testset "Testing stochastic fictitious play" begin

        normal = Normal()  #standard normal distribution

        sfp_dec = StochasticFictitiousPlay(g, normal)
        sfp_con = StochasticFictitiousPlay(g, normal, ConstantGain(gain))

        x_dec = play(sfp_dec, init_actions)
        @test sum(x_dec[1]) == 1
        @test sum(x_dec[2]) == 1
        x_con = play(sfp_con, init_actions)
        @test sum(x_con[1]) == 1
        @test sum(x_con[2]) == 1

        y_dec = time_series(sfp_dec, 3, init_actions)
        for t in 1:3
            @test sum(y_dec[1][:,t]) == 1
            @test sum(y_dec[2][:,t]) == 1
        end
        y_con = time_series(sfp_con, 3, init_actions)
        for t in 1:3
            @test sum(y_con[1][:,t]) == 1
            @test sum(y_con[2][:,t]) == 1
        end
    end
end