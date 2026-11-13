"""Tests for EngineDriver: advance(), decider() and apply(action), on both engines."""
import random

import pytest

from actions import Discard, EndTablePlay, GoDown, Pass, PickStock, Place, Wait
from cards import Card, Rank, Suit
from engine import JoeEngine as SlowEngine
from fast_engine import JoeEngine as FastEngine
from game_context import GameContext

ENGINES = [pytest.param(FastEngine, id="fast_engine"), pytest.param(SlowEngine, id="engine")]


def new_game(engine_cls, num_players=4, seed=0):
    random.seed(seed)
    ctx = GameContext(num_players=num_players)
    engine = engine_cls(ctx)
    engine.advance()
    return ctx, engine


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_advance_reaches_the_first_pickup(engine_cls):
    ctx, engine = new_game(engine_cls)
    assert engine.decider() == ('pickup_decision', (ctx.dealer_idx + 1) % 4)
    assert len(ctx.players[0].hand_list) == 11


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_the_may_i_decision_belongs_to_the_player_offered_it(engine_cls):
    ctx, engine = new_game(engine_cls)
    active = ctx.active_player_idx
    engine.apply(PickStock())
    assert engine.decider() == ('may_i_decision', (active + 1) % 4)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_apply_moves_on_to_the_next_decision(engine_cls):
    ctx, engine = new_game(engine_cls)
    first = ctx.active_player_idx
    engine.apply(PickStock())
    while engine.decider()[0] == 'may_i_decision':
        engine.apply(Pass())
    state, idx = engine.decider()
    if state == 'go_down_decision':
        engine.apply(Wait())
        state, idx = engine.decider()
    assert (state, idx) == ('discard_phase', first)

    card = ctx.players[first].hand_list[0]
    engine.apply(Discard(card.suit, card.rank))

    assert engine.decider() == ('pickup_decision', (first + 1) % 4)


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_apply_refuses_an_illegal_action_and_changes_nothing(engine_cls):
    ctx, engine = new_game(engine_cls)
    decision = engine.decider()
    hand = list(ctx.active_player.hand_list)
    card = hand[0]
    with pytest.raises(ValueError):
        engine.apply(Discard(card.suit, card.rank))  # not a pickup action
    assert engine.decider() == decision
    assert ctx.active_player.hand_list == hand


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_go_down_accepts_the_melds_in_any_order(engine_cls):
    ctx, engine = new_game(engine_cls)
    player = ctx.active_player
    player.hand_list.clear()
    player.private_hand.fill(0)
    player.receive_cards([Card(s, r) for r in (Rank.SEVEN, Rank.NINE)
                          for s in (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS)]
                         + [Card(Suit.CLUBS, Rank.TWO)])
    engine.apply(PickStock())
    while engine.decider()[0] == 'may_i_decision':
        engine.apply(Pass())

    engine.apply(GoDown((('set', Rank.NINE), ('set', Rank.SEVEN))))

    assert player.is_down
    assert engine.decider()[0] == 'table_play_phase'


def play_simply(engine_cls, seed):
    """
    Plays a full game where each player goes down and places cards whenever
    possible, never calls a May-I, and otherwise picks at random.
    Returns the context and the finished engine.
    """
    ctx, engine = new_game(engine_cls, seed=seed)
    rng = random.Random(seed)
    while (decision := engine.decider()) is not None:
        state, idx = decision
        legal = ctx.legal_actions(idx, state)
        eager = [a for a in legal if isinstance(a, (GoDown, Place))]
        if eager:
            action = eager[0]
        elif state == 'table_play_phase':
            action = EndTablePlay()
        elif state == 'may_i_decision':
            action = Pass()
        else:
            action = rng.choice(legal)
        engine.apply(action)
    return ctx, engine


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_apply_refuses_when_the_game_is_over(engine_cls):
    ctx, engine = play_simply(engine_cls, seed=0)
    assert engine.state_id == 'game_over'
    assert engine.decider() is None
    with pytest.raises(ValueError):
        engine.apply(PickStock())


def test_both_engines_play_identical_games_through_apply():
    fast_ctx, _ = play_simply(FastEngine, seed=3)
    slow_ctx, _ = play_simply(SlowEngine, seed=3)
    assert [p.score for p in fast_ctx.players] == [p.score for p in slow_ctx.players]
