abstract type AbstractGain end

struct DecreasingGain <: AbstractGain end

mutable struct ConstantGain{T<:Real} <: AbstractGain
    size::T
end

step_size(T::Type, gain::DecreasingGain, t::Integer) = one(T)/(t+1)
step_size(T::Type, gain::ConstantGain, t::Integer) = T(gain.size)


abstract type AbstractFictitiousPlay{N,T<:Real} end

struct FictitiousPlay{N,T<:Real,TG<:AbstractGain} <: AbstractFictitiousPlay{N,T}
    players::NTuple{N,Player{N,T}}
    nums_actions::NTuple{N,Int}
    gain::TG

    function FictitiousPlay{N,T,TG}(g::NormalFormGame{N,T},
                                    gain::TG) where {N,T<:Real,TG<:AbstractGain}
        return new(g.players, g.nums_actions, gain)
    end
end

FictitiousPlay(g::NormalFormGame{N,T},
               gain::TG) where {N,T<:Real,TG<:AbstractGain} =
    FictitiousPlay{N,T,TG}(g, gain)
FictitiousPlay(g::NormalFormGame) = FictitiousPlay(g, DecreasingGain())

function play!(fp::FictitiousPlay{N},
               actions::MixedActionProfile{TA,N},
               options::BROptions,
               brs::Vector{Int}, t::Integer) where {N,TA<:Real}
    for i in 1:N
        opponents_actions =
            tuple(actions[i+1:end]..., actions[1:i-1]...)
        brs[i] = best_response(fp.players[i], opponents_actions, options)
    end

    for i in 1:N
        actions[i] .*= 1 - step_size(TA, fp.gain, t)
        actions[i][brs[i]] += step_size(TA, fp.gain, t)
    end

    return actions
end

struct StochasticFictitiousPlay{N,T<:Real,TG<:AbstractGain,
                                TD<:Distribution} <: AbstractFictitiousPlay{N,T}
    players::NTuple{N,Player{N,T}}
    nums_actions::NTuple{N,Int}
    gain::TG
    d::TD

    function StochasticFictitiousPlay{N,T,TG,TD}(
        g::NormalFormGame{N,T}, gain::TG, d::TD
    ) where {N,T<:Real,TG<:AbstractGain,TD<:Distribution}
        return new(g.players, g.nums_actions, gain, d)
    end
end

StochasticFictitiousPlay(
    g::NormalFormGame{N,T}, gain::TG, d::TD
) where {N,T<:Real,TG<:AbstractGain,TD<:Distribution} =
    StochasticFictitiousPlay{N,T,TG,TD}(g, gain, d)
StochasticFictitiousPlay(g::NormalFormGame, d::Distribution) =
    StochasticFictitiousPlay(g, DecreasingGain(), d)

function play!(fp::StochasticFictitiousPlay{N},
               actions::MixedActionProfile{TA,N},
               options::BROptions,
               brs::Vector{Int}, t::Integer) where {N,TA<:Real}
    for i in 1:N
        opponents_actions =
            tuple(actions[i+1:end]..., actions[1:i-1]...)
        perturbations = rand(fp.d, fp.nums_actions[i])
        brs[i] = best_response(fp.players[i], opponents_actions, perturbations)
    end

    for i in 1:N
        actions[i] .*= 1 - step_size(TA, fp.gain, t)
        actions[i][brs[i]] += step_size(TA, fp.gain, t)
    end

    return actions
end

function play!(fp::AbstractFictitiousPlay{N},
               actions::MixedActionProfile{TA,N},
               options::BROptions=BROptions();
               num_reps::Integer=1, t_init::Integer=1) where {N,TA<:Real}
    brs = Vector{Int}(undef, N)
    for t in t_init:(t_init+num_reps-1)
        play!(fp, actions, options, brs, t)
    end
    return actions
end

function play(fp::AbstractFictitiousPlay{N},
              actions::MixedActionProfile{TA,N},
              options::BROptions=BROptions();
              num_reps::Integer=1, t_init::Integer=1) where {N,TA<:Real}
    Tout = typeof(zero(TA)/one(TA))
    actions_copied::NTuple{N,Vector{Tout}} =
        ntuple(i -> copyto!(similar(actions[i], Tout), actions[i]), N)
    play!(fp, actions_copied, options, num_reps=num_reps, t_init=t_init)
end

function play(fp::AbstractFictitiousPlay{N},
              actions::PureActionProfile{N},
              options::BROptions=BROptions();
              num_reps::Integer=1, t_init::Integer=1) where {N}
    mixed_actions = ntuple(i -> pure2mixed(fp.nums_actions[i], actions[i]), N)
    play!(fp, mixed_actions, options, num_reps=num_reps, t_init=t_init)
end

function play(fp::AbstractFictitiousPlay{N},
              options::BROptions=BROptions();
              num_reps::Integer=1, t_init::Integer=1) where {N}
    play!(fp, random_mixed_actions(fp.nums_actions), options,
          num_reps=num_reps, t_init=t_init)
end

function time_series!(fp::AbstractFictitiousPlay{N},
                      out::NTuple{N,Matrix{<:Real}},
                      options::BROptions=BROptions();
                      t_init::Integer=1) where {N}
    ts_length = size(out[1], 2)
    actions = ntuple(i -> out[i][:, 1], N)
    brs = Vector{Int}(undef, N)

    for j in 2:ts_length
        play!(fp, actions, options, brs, t_init - 1 + j - 1)
        for i in 1:N
            out[i][:, j] = actions[i]
        end
    end

    return out
end

function _copy_action_to!(dest::AbstractVector, src::MixedAction)
    dest[:] = src
    return dest
end

function _copy_action_to!(dest::AbstractVector, src::PureAction)
    dest .= 0
    dest[src] = 1
    return dest
end

for (ex_TAS, ex_where, ex_T) in (
        (:(MixedActionProfile{TA,N}), (:(N), :(T<:Real), :(TA<:Real)), :(TA)),
        (:(PureActionProfile{N}), (:(N), :(T<:Real)), :(T))
    )
    @eval function time_series(fp::AbstractFictitiousPlay{N,T},
                               ts_length::Integer,
                               init_actions::$(ex_TAS),
                               options::BROptions=BROptions();
                               t_init::Integer=1) where $(ex_where...)
        Tout = typeof(zero($(ex_T))/one($(ex_T)))
        out::NTuple{N,Matrix{Tout}} =
            ntuple(i -> Matrix{Tout}(undef, fp.nums_actions[i], ts_length), N)
        for i in 1:N
            _copy_action_to!(@views(out[i][:, 1]), init_actions[i])
        end
        time_series!(fp, out, options, t_init=t_init)
    end
end

function time_series(fp::AbstractFictitiousPlay{N},
                     ts_length::Integer,
                     options::BROptions=BROptions();
                     t_init::Integer=1) where {N}
    time_series(fp, ts_length, random_mixed_actions(fp.nums_actions), options,
                t_init=t_init)
end
