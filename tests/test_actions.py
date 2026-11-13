"""Tests for the action types (actions.py) and GameContext.legal_actions()."""
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
from cards import Card, Rank, Suit
from game_context import GameContext

S, H, C, D = Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS


def ctx_with_hand(cards, round_idx=0):
    ctx = GameContext(num_players=4)
    ctx.current_round_idx = round_idx
    player = ctx.players[0]
    player.hand_list.clear()
    player.private_hand.fill(0)
    player.receive_cards(cards)
    return ctx


def test_pickup_offers_the_stock_and_an_available_discard():
    ctx = GameContext(num_players=4)
    assert ctx.legal_actions(0, 'pickup_decision') == [PickStock()]  # empty discard pile
    ctx.discard_pile.append(Card(S, Rank.ACE))
    assert ctx.legal_actions(0, 'pickup_decision') == [PickStock(), PickDiscard()]


def test_may_i_offers_call_or_pass():
    assert GameContext().legal_actions(1, 'may_i_decision') == [CallMayI(), Pass()]


def test_discard_offers_only_cards_in_hand():
    ctx = ctx_with_hand([Card(H, Rank.TWO), Card(C, Rank.KING)])
    assert ctx.legal_actions(0, 'discard_phase') == [Discard(H, Rank.TWO), Discard(C, Rank.KING)]


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


def test_go_down_offers_only_wait_without_the_objective():
    ctx = ctx_with_hand([Card(S, Rank.SEVEN), Card(H, Rank.SEVEN), Card(D, Rank.NINE)])
    assert ctx.legal_actions(0, 'go_down_decision') == [Wait()]


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
