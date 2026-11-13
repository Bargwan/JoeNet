"""Tests for the action types (actions.py) and GameContext.legal_actions()."""
import random
from collections import Counter

import numpy as np
import pytest

from actions import (
    CallMayI,
    Destination,
    Discard,
    EndTablePlay,
    GoDown,
    Pass,
    PickDiscard,
    PickStock,
    Place,
    Wait,
)
from agents import HeuristicAgent
from cards import Card, Rank, Suit
from evaluate_arena import apply_engine_action
from fast_engine import JoeEngine
from game_context import (
    CALL_MAY_I,
    END_TABLE_PLAY,
    GO_DOWN,
    PASS,
    PICK_DISCARD,
    PICK_STOCK,
    WAIT,
    GameContext,
    discard_action,
    place_action,
)

S, H, C, D = Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS


def action_number(action, state_id):
    """The mask number the old numbered scheme gives an action."""
    if isinstance(action, Place):
        return place_action(action.suit, action.rank, action.destination)
    if isinstance(action, Discard):
        return discard_action(action.suit, action.rank)
    if isinstance(action, (Wait, EndTablePlay)):
        return WAIT if state_id == 'go_down_decision' else END_TABLE_PLAY
    return {PickStock: PICK_STOCK, PickDiscard: PICK_DISCARD, CallMayI: CALL_MAY_I,
            Pass: PASS, GoDown: GO_DOWN}[type(action)]


@pytest.mark.parametrize("num_players, seed", [(4, 0), (3, 2)])
def test_legal_actions_match_the_action_mask_throughout_a_game(num_players, seed):
    """At every decision of full games, legal_actions allows exactly what the mask allows."""
    random.seed(seed)
    ctx = GameContext(num_players=num_players)
    engine = JoeEngine(ctx)
    agents = [HeuristicAgent(random_seed=i) for i in range(num_players)]
    checked = Counter()

    engine.start_game()
    while engine.state_id != 'game_over':
        state = engine.state_id
        if state == 'dealing':
            engine.deal_cards()
            continue
        if state == 'start_turn':
            engine.enter_pickup()
            continue
        idx = ctx.may_i_target_idx if state == 'may_i_decision' else ctx.active_player_idx

        mask = ctx.get_action_mask(idx, state)
        legal = ctx.legal_actions(idx, state)
        assert {action_number(a, state) for a in legal} == set(np.flatnonzero(mask)), state
        assert len(legal) == len(set(legal)), f"duplicate legal actions in {state}"
        checked[state] += 1

        action = agents[idx].select_action(state, ctx, idx, mask)
        apply_engine_action(engine, ctx, state, idx, action, agents[idx])

    assert set(checked) == {'pickup_decision', 'may_i_decision', 'go_down_decision',
                            'table_play_phase', 'discard_phase'}


def ctx_with_hand(cards, round_idx=0):
    ctx = GameContext(num_players=4)
    ctx.current_round_idx = round_idx
    player = ctx.players[0]
    player.hand_list.clear()
    player.private_hand.fill(0)
    player.receive_cards(cards)
    return ctx


def test_each_card_is_listed_once_however_many_copies_are_held():
    ctx = ctx_with_hand([Card(S, Rank.SEVEN, 0), Card(S, Rank.SEVEN, 1), Card(H, Rank.TWO)])
    assert ctx.legal_actions(0, 'discard_phase') == [Discard(S, Rank.SEVEN), Discard(H, Rank.TWO)]


def test_go_down_lists_each_way_to_go_down_plus_wait():
    hand = [Card(s, r) for r in (Rank.SEVEN, Rank.NINE, Rank.QUEEN) for s in (S, H, D)]
    ctx = ctx_with_hand(hand)  # [3, 3] with three possible sets
    legal = ctx.legal_actions(0, 'go_down_decision')
    assert legal[-1] == Wait()
    assert sorted(a.melds for a in legal[:-1]) == sorted([
        (('set', Rank.SEVEN), ('set', Rank.NINE)),
        (('set', Rank.SEVEN), ('set', Rank.QUEEN)),
        (('set', Rank.NINE), ('set', Rank.QUEEN)),
    ])


def test_table_play_lists_every_destination_for_each_key_card():
    from game_context import TableRun
    ctx = ctx_with_hand([Card(S, Rank.SEVEN), Card(H, Rank.KING)])
    ctx.players[0].is_down = True
    for suit in (H, C, D):
        ctx.table_sets[suit, Rank.SEVEN] = 1
    ctx.runs_on_table.append(TableRun(S, Rank.THREE, Rank.SIX))
    assert ctx.legal_actions(0, 'table_play_phase') == [
        Place(S, Rank.SEVEN, Destination.ON_SET),
        Place(S, Rank.SEVEN, Destination.RUN_HIGH_END),
        EndTablePlay(),
    ]


def test_an_exposed_discard_cannot_be_picked_up():
    ctx = GameContext(num_players=4)
    ctx.discard_pile[:] = [Card(S, Rank.TWO), Card(S, Rank.THREE)]
    assert PickDiscard() in ctx.legal_actions(1, 'pickup_decision')
    ctx.active_player_idx = 1
    ctx.execute_pickup_discard()
    assert ctx.legal_actions(1, 'pickup_decision') == [PickStock()]


def test_unknown_state_is_refused():
    with pytest.raises(ValueError):
        GameContext().legal_actions(0, 'dealing')


def test_actions_compare_and_hash_by_value():
    assert Discard(S, Rank.SEVEN) == Discard(S, Rank.SEVEN)
    assert len({Discard(S, Rank.SEVEN), Discard(S, Rank.SEVEN), PickStock(), PickStock()}) == 2


@pytest.mark.parametrize("action, text", [
    (PickStock(), "Pick up from the stock"),
    (Place(S, Rank.SEVEN, Destination.RUN_HIGH_END), "Place 7♠ on the high end of a run"),
    (Discard(H, Rank.ACE), "Discard A♡"),
    (GoDown((('run', S, 10), ('set', Rank.NINE))), "Go down with run J♠-A♠, set of Nines"),
])
def test_actions_read_naturally(action, text):
    assert str(action) == text
