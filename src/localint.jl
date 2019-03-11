#= 
Tools for local interaction model

=#


# LocalInteraction

"""

	LocalInteraction{N, T, S}

Type representing the local interaction model with N players.

# Fields

- `players::NTuple{N,Player{2,T}}` : Tuple of player instances.
- `num_actions::Integer` : The number of actions for players.
- `adj_matrix::Array{S,2}` : Adjacency matrix of the graph in the model.
"""

struct LocalInteraction{N,T<:Real,S<:Integer}
	players::NTuple{N,Player{2,T}}
	num_actions::Integer
	adj_matrix::Array{S,2}

	function LocalInteraction(g::NormalFormGame{2,T},
														adj_matrix::Array{S,2}) where {T<:Real,S<:Int}
		N = size(adj_matrix, 1)
		players = ntuple(i -> g.players[1], N)
		return new{N,T,S}(players, g.nums_actions[1], adj_matrix)
	end
end

LocalInteraction(g::NormalFormGame{2,<:Real}, adj_matrix::Array{<:Integer,2}) =
	LocalInteraction(g, adj_matrix)


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

play!(li::LocalInteraction, actions::Vector{Int}, options::BROptions, 
																									player_ind::Int) = 
	play!(li, actions, options, [player_ind])

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
							 actions::Vector{Int},
							 options::BROptions,
							 player_ind::Union{Vector{Int},Integer},
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
							player_ind::Union{Vector{Int},Integer},
							num_reps::Integer=1,
							options::BROptions=BROptions()) where N
	actions_vector = [i for i in actions]
	actions_vector = play!(li, actions_vector, options, player_ind, num_reps)
	new_actions = ntuple(i -> actions_vector[i], N)
	return new_actions
end

function play(li::LocalInteraction,
							player_ind::Union{Vector{Int},Integer},
							num_reps::Integer=1,
							options::BROptions=BROptions())
	nums_actions = ntuple(i -> li.num_actions, N)
	play(li, random_pure_actions(nums_actions), player_ind, num_reps, options)
end

function play(li::LocalInteraction{N},
							actions::PureActionProfile,
							num_reps::Integer=1,
							options::BROptions=BROptions()) where N
	play(li, actions, [1:N...], num_reps, options)
end

function play(li::LocalInteraction{N},
							options::BROptions=BROptions();
							num_reps::Integer=1) where N
	nums_actions = ntuple(i -> li.num_actions, N)
	play(li, random_pure_actions(nums_actions), [1:N...], num_reps, options)
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
											out::Matrix{Int},
											options::BROptions,
											player_ind::Union{Vector{Int},Integer}) where N
	ts_length = size(out, 2)
	actions = [out[i,1] for i in 1:N]

	for t in 2:ts_length
		play!(li, actions, options, player_ind)
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

function time_series(li::LocalInteraction{N},
										 ts_length::Integer,
										 init_actions::PureActionProfile,
										 player_ind::Union{Vector{Int},Integer},
										 options::BROptions=BROptions()) where N
	out = Matrix{Int}(undef, N, ts_length)
	for i in 1:N
		out[i,1] = init_actions[i]
	end
	time_series!(li, out, options, player_ind)
end

function time_series(li::LocalInteraction{N},
										 ts_length::Integer,
										 init_actions::PureActionProfile,
										 options::BROptions=BROptions()) where N
	time_series(li, ts_length, init_actions, [1:N...], options)
end

function time_series(li::LocalInteraction,
										 ts_length::Integer,
										 player_ind::Union{Vector{Int},Integer},
										 options::BROptions=BROptions())
	nums_actions = ntuple(i -> li.num_actions, N)
	time_series(li, ts_length, random_pure_actions(nums_actions), player_ind, options)
end

function time_series(li::LocalInteraction{N},
										 ts_length::Integer,
										 options::BROptions=BROptions()) where N
	nums_actions = ntuple(i -> li.num_actions, N)
	time_series(li, ts_length, random_pure_actions(nums_actions), [1:N...], options)
end