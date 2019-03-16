#= 
Tools for local interaction model

=#

using SparseArrays


# LocalInteraction

"""

　　LocalInteraction{N, T, S}

Type representing the local interaction model with N players.

# Fields

- `players::NTuple{N,Player{2,T}}` : Tuple of player instances.
- `num_actions::Integer` : The number of actions for players.
- `adj_matrix::Array{S,2}` : Adjacency matrix of the graph in the model.
"""

struct LocalInteraction{N,T<:Real,S<:Real}
    players::NTuple{N,Player{2,T}}
    num_actions::Int
    adj_matrix::SparseMatrixCSC{S,<:Integer}

    function LocalInteraction(g::NormalFormGame{2,T},
                              adj_matrix::Matrix{S}) where {T<:Real,S<:Real}
        N = size(adj_matrix, 1)
        players = ntuple(i -> g.players[1], N)
        sparse_adj = sparse(adj_matrix)
        return new{N,T,S}(players, g.nums_actions[1], sparse_adj)
    end
end


# play!

function _vector_to_matrix(li::LocalInteraction{N}, actions::Vector{Int}) where N
    matrix_action = zeros(Int, N, li.num_actions)
    for (i, action) in enumerate(actions)
        matrix_action[i, action] = 1
    end
    return matrix_action
end

"""

    play!(li, actions, options, player_ind)

Update `actions` given adjacency matrix and actions of each players.

# Arguments

- `li::LocalInteraction{N}` : Local interaction instance.
- `actions::Vector{Int}` : Vector of actions of each players.
- `options::BROptions` : Options for `best_response` method.
- `player_ind::Vector{Int}` : Vector of integers representing the index of
    players to take an action.

# Returns

- `actions::Vector{Int}` : Updated `actions`. 
"""

function play!(li::LocalInteraction{N},
               actions::Vector{Int},
               options::BROptions,
               player_ind::Vector{Int}) where N
    actions_matrix = _vector_to_matrix(li, actions)
    opponent_action = li.adj_matrix[player_ind,:] * actions_matrix
    for (k, i) in enumerate(player_ind)
        br = best_response(li.players[i], opponent_action[k,:], options)
        actions[i] = br
    end
    return actions
end

play!(li::LocalInteraction, actions::Vector{<:Integer}, options::BROptions, 
    player_ind::Int) = play!(li, actions, options, [player_ind])

play!(li::LocalInteraction{N}, actions::Vector{<:Integer}, options::BROptions) where {N} =
    play!(li, actions, options, [1:N...])

"""

    play!(li, actions, options, player_ind, num_reps)

Update actions of each players `num_reps` times.

# Arguments

- `li::LocalInteraction` : Local interaction instance.
- `actions::Vector{Int}` : Vector of actions of each players.
- `options::BROptions` : Options for `best_response` method.
- `player_ind::Union{Vector{Int},Integer} : Integer or vector of integers
    representing the index of players to take an action.
- `num_reps::Integer` : The number of iterations.

# Returns

- `actions::Vector{Int}` : Updated `actions`.
"""

function play!(li::LocalInteraction,
               actions::Vector{<:Integer},
               options::BROptions,
               player_ind::Union{Vector{<:Integer},Integer},
               num_reps::Integer=1)
    for t in 1:num_reps
        play!(li, actions, options, player_ind)
    end
    return actions
end


# play

"""

    play(li, actions, player_ind, num_reps, options)

Return the actions of each players after `num_reps` times iteration.

# Arguments

- `li::LocalInteraction{N}` : Local interaction instance.
- `actions::PureActionProfile` : Initial actions of each players.
- `player_ind::Union{Vector{Int},Integer}` : Integer or vector of integers
	representing the index of players to take an action.
- `num_reps::Integer` : The number of iterations.
- `options::BROptions` : Options for `best_response` method.

# Returns

- `::PureActionProfile` : Actions of each players after iterations.
"""

function play(li::LocalInteraction{N},
              actions::PureActionProfile,
              player_ind::Union{Vector{<:Integer},Integer},
              options::BROptions=BROptions();
              num_reps::Integer=1) where N
    actions_vector = [i for i in actions]
    actions_vector = play!(li, actions_vector, options, player_ind, num_reps)
    new_actions = ntuple(i -> actions_vector[i], N)
    return new_actions
end

function play(li::LocalInteraction{N},
              actions::PureActionProfile,
              options::BROptions=BROptions();
              num_reps::Integer=1) where N
    play(li, actions, [1:N...], options, num_reps=num_reps)
end


# time_series!

"""

    time_series!(li, out, options, player_ind)

Update `out` which is time series of actions.

# Arguments

- `li::LocalInteraction{N}` : Local interaction instance.
- `out::Matrix{Int}` : Matrix representing time series of actions of each players.
- `options::BROptions` : Options for `best_response` method.
- `player_ind::Union{Vector{Int},Integer}` : Integer or vector of integers
    representing the index of players to take an action.

# Returns

- `out::Matrix{Int}` : Updated `out`.
"""

function time_series!(li::LocalInteraction{N},
                      out::Matrix{<:Integer},
                      options::BROptions,
                      player_ind_seq::Vector{<:Any}) where N
    ts_length = size(out, 2)
    if ts_length != length(player_ind_seq) + 1
        throw(ArgumentError("The length of `ts_length` and `player_ind_seq` are mismatched"))
    end

    actions = [out[i,1] for i in 1:N]
    for t in 2:ts_length
        play!(li, actions, options, player_ind_seq[t-1])
        out[:,t] = actions
    end

    return out
end

function time_series!(li::LocalInteraction{N},
                      out::Matrix{<:Integer},
                      options::BROptions) where N
    ts_length = size(out, 2)
    actions = [out[i,1] for i in 1:N]
    for t in 2:ts_length
        play!(li, actions, options)
        out[:,t] = actions
    end
    
    return out
end


# time_series

"""

    time_series(li, ts_length, init_actions, player_ind, options)

Return the time series of actions.

# Arguments

- `li::LocalInteraction{N}` : Local interaction instance.
- `ts_length::Integer` : The length of time series.
- `init_actions::PureActionProfile` : Initial actions of iterations.
- `player_ind::Union{Vector{Int},Integer}` : Integer or vector of integers
    representing the index of players to take an action.
- `options::BROptions` : Options for `best_response` method.

# Returns

- `::Matrix{Int}` : Time series of players' actions.
"""

#deterministic(sequencial)
function time_series(li::LocalInteraction{N},
                     ts_length::Integer,
                     init_actions::PureActionProfile,
                     player_ind_seq::Vector{<:Any},
                     options::BROptions=BROptions()) where N
    out = Matrix{Int}(undef, N, ts_length)
    for i in 1:N
        out[i,1] = init_actions[i]
    end
    time_series!(li, out, options, player_ind_seq)
end

# simultaneous and random
function time_series(li::LocalInteraction{N},
                     ts_length::Integer,
                     init_actions::PureActionProfile,
                     options::BROptions=BROptions();
                     revision::Symbol=:simultaneous) where N
    if revision == :simultaneous
        out = Matrix{Int}(undef, N, ts_length)
        for i in 1:N
            out[i,1] = init_actions[i]
        end
        time_series!(li, out, options)
    elseif revision == :random
        player_ind_seq = rand(1:N, ts_length-1)
        time_series(li, ts_length, init_actions, player_ind_seq, options)
    else
        throw(ArgumentError("revision argument must be `simultaneous` or `random`"))
    end
end

function time_series(li::LocalInteraction{N},
                     ts_length::Integer,
                     player_ind_seq::Vector{<:Any},
                     options::BROptions=BROptions()) where N
    nums_actions = ntuple(i -> li.num_actions, N)
    actions = random_pure_actions(nums_actions)
    time_series(li, ts_length, actions, player_ind_seq, options)
end

function time_series(li::LocalInteraction{N},
                     ts_length::Integer,
                     options::BROptions=BROptions();
                     revision::Symbol=:simultaneous) where N
    nums_actions = ntuple(i -> li.num_actions, N)
    actions = random_pure_actions(nums_actions)
    time_series(li, ts_length, actions, options, revision=revision)
end