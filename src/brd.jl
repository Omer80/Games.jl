#=
    Tools for best response dynamics

=#

using Sort
using StatsBase

abstract type AbstractBRD{T<:Real} end

struct BRD{T<:Real} <: AbstractBRD{T<:Real}
    N::Int
    player::Player{2,T}
    num_actions::Int
end

BRD(N<:Integer, payoff_array::Matrix{T}) where {T<:Real} =
    BRD(N, Player(payoff_array), size(payoff_array, 1))

struct KMR{T<:Real} <: AbstractBRD{T<:Real}
    N::Int
    player::Player{2,T}
    num_actions::Int
    epsilon::Float
end

KMR(N<:Integer, payoff_array::Matrix{T}, epsilon<:Real) where {T<:Real} =
    KMR(N, Player(payoff_array), size(payoff_array, 1), epsilon)

struct SamplingBRD{T<:Real} <: AbstractBRD{T<:Real}
    N::Int
    player::Player{2,T}
    num_actions::Int
    k::Int  #sample size
end

SamplingBRD(N<:Integer, payoff_array::Matrix{T}, k::Int) where {T<:Real} =
    SamplingBRD(N, Player(payoff_array), size(payoff_array, 1), k)

function set_action_dist(abrd::AbstractBRD, actions::PureActionProfile)
    if abrd.N != length(actions)
        throw(ArgumentError("The length of action profile must
                             equal to the number of players"))
    end
    action_dist = zeros(abrd.num_actions)
    for i in 1:abrd.N
        action_dist[actions[i]] += 1
    end
    return action_dist
end

function set_action_dist(abrd::AbstractBRD)
    nums_actions = ntuple(i -> abrd.num_actions, abrd.N)
    actions = random_pure_actions(nums_actions)
    return set_action_dist(brs, actions)
end

for (i, brd) in ((1, :BRD), (2, :KMR), (3, :SamplingBRD))
    @eval function play!(brd::$(brd),
                         action<:Integer,
                         action_dist::Vector{<:Integer},
                         options::BROptions)
        action_dist[action] -= 1
        if i == 1
            next_action = best_response(brd.player, actions, options)
        elseif i == 2
            if rand() <= brd.epsilon
                next_action = rand(1:brd.num_actions)
            else
                next_action = best_response(brd.player, actions, options)
            end
        else
            actions = sample(1:brd.num_actions, Weights(action_dist), brd.k)
            sample_action_dist = zeros(brd.num_actions, dtype=Int)
            for a in actions
                sample_action_dist[a] += 1
            end
            next_action = best_response(brd.player, sample_action_dist, options)
        end
        action_dist[next_action] += 1
        return action_dist
    end
end

function time_series!(abrd::AbstractBRD,
                      out::Matrix{<:Integer},
                      options::BROptions)
    ts_length = size(out, 1)
    player_ind_seq = rand(1:abrd.N, ts_length)
    action_dist = [out[i,1] for i in 1:abrd.num_actions]
    for t in 1:ts_length
        action = searchsortedlast(accumulate(+, action_dist), player_ind_seq[t])
        action_dist = play!(abrd, action, action_dist, options)
        for i in 1:brd.num_actions
            out[i,t+1] = action_dist[i]
        end
    end
    return out
end

function time_series(abrd::AbstractBRD,
                     ts_length<:Integer,
                     init_actions::Union{PureActionProfile,Nothing}=nothing,
                     options::BROptions=BROptions())
    player_ind_seq = rand(1:abrd.N, ts_length)
    action_dist = set_action_dist(abrd, init_actions)
    out = Matrix{Int}(undef, brd.num_actions, ts_length)
    for i in 1:abrd.num_actions
        out[i,1] = action_dist[i]
    end
    time_series!(abrd, out, options)
end