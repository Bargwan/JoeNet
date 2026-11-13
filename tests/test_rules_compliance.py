"""
Rules-compliance tests: each test states a rule from rules.md and checks the
engine obeys it.

The tests only use public behaviour (going down, the table-play action mask,
engine transitions, card movement), never the internal table representation,
so they keep working when that representation changes.

Tests for known, not-yet-fixed deviations are marked xfail(strict=True). When
a fix lands, its test starts passing, strict mode turns that into a failure,
and the fix must remove the marker.
"""
import random

import pytest

from cards import Card, Rank, Suit
from engine import JoeEngine as SlowEngine
from fast_engine import JoeEngine as FastEngine
from game_context import (
    GO_DOWN,
    ON_SET,
    RUN_HIGH_END,
    RUN_LOW_END,
    GameContext,
    place_action,
)
from tests.test_utils import verify_mass_integrity

ENGINES = [pytest.param(FastEngine, id="fast_engine"), pytest.param(SlowEngine, id="engine")]

S, H, C, D = Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS
A, K, Q, J = Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK


def run(suit, *ranks):
    return [Card(suit, r) for r in ranks]


def ranks(lo, hi):
    """Rank range by face value, e.g. ranks(3, 6) -> THREE..SIX."""
    return [Rank(v - 1) for v in range(lo, hi + 1)]


# A set of 9s in non-spade suits, used to fill a round objective that needs a set.
FILLER_SET_OF_9S = [Card(H, Rank.NINE), Card(C, Rank.NINE), Card(D, Rank.NINE)]


def table_with(melds_hand, round_idx):
    """
    Lays melds on the table the legitimate way: player 0 holds exactly the
    objective cards for `round_idx` and goes down with them.
    """
    ctx = GameContext(num_players=4)
    ctx.current_round_idx = round_idx
    player = ctx.players[0]
    player.hand_list.clear()
    player.private_hand.fill(0)
    player.receive_cards(melds_hand)

    assert ctx.check_hand_objective(0), "test setup: hand does not meet the objective"
    player.is_down = True
    ctx.go_down(0)
    assert player.hand_list == [], "test setup: going down did not lay every card"
    return ctx


def offered_destinations(ctx, card, player_idx=0):
    """The destinations (ON_SET, RUN_LOW_END, RUN_HIGH_END) the engine offers for `card`."""
    player = ctx.players[player_idx]
    player.receive_cards([card])
    mask = ctx.get_action_mask(player_idx, "table_play_phase")
    player.discard(int(card.suit), int(card.rank))
    return {d for d in (ON_SET, RUN_LOW_END, RUN_HIGH_END)
            if mask[place_action(card.suit, card.rank, d)]}


def can_place(ctx, card, player_idx=0):
    """True if the engine offers `card` as a legal table play anywhere."""
    return bool(offered_destinations(ctx, card, player_idx))


def place(ctx, card, destination, player_idx=0):
    """Places `card` from the player's hand at `destination`, as the engine would."""
    player = ctx.players[player_idx]
    player.receive_cards([card])
    ctx.execute_table_play(player_idx, len(player.hand_list) - 1, destination)


def xfail(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


# ---------------------------------------------------------------------------
# Section 6: Placing Key Cards
# ---------------------------------------------------------------------------

def table_with_spade_run_3_to_6():
    # Round 2 [3, 4]: one set (the filler 9s) and the run ♠3-4-5-6.
    return table_with(run(S, *ranks(3, 6)) + FILLER_SET_OF_9S, round_idx=1)


@pytest.mark.parametrize("card", [Card(S, Rank.TWO), Card(S, Rank.SEVEN)], ids=str)
def test_run_accepts_cards_at_either_end(card):
    ctx = table_with_spade_run_3_to_6()
    assert can_place(ctx, card)


@pytest.mark.parametrize(
    "card",
    [
        pytest.param(Card(S, Rank.FOUR), id="4♠ (middle)"),
        pytest.param(Card(S, Rank.FIVE), id="5♠ (middle)"),
        pytest.param(Card(S, Rank.THREE), id="3♠ (duplicate of low end)"),
        pytest.param(Card(S, Rank.SIX), id="6♠ (duplicate of high end)"),
    ],
)
def test_run_rejects_cards_that_do_not_extend_an_end(card):
    ctx = table_with_spade_run_3_to_6()
    assert not can_place(ctx, card)


def test_run_rejects_other_suits():
    ctx = table_with_spade_run_3_to_6()
    assert not can_place(ctx, Card(H, Rank.SEVEN))


def table_with_high_ace_run():
    # ♠J-Q-K-A
    return table_with(run(S, J, Q, K, A) + FILLER_SET_OF_9S, round_idx=1)


def test_high_ace_run_takes_the_ten():
    assert can_place(table_with_high_ace_run(), Card(S, Rank.TEN))


def test_high_ace_run_does_not_wrap_to_two():
    assert not can_place(table_with_high_ace_run(), Card(S, Rank.TWO))


@pytest.mark.parametrize("card", [Card(S, A), Card(S, K)], ids=str)
def test_high_ace_run_rejects_duplicates(card):
    assert not can_place(table_with_high_ace_run(), card)


def table_with_low_ace_run():
    # ♠A-2-3-4
    return table_with(run(S, A, *ranks(2, 4)) + FILLER_SET_OF_9S, round_idx=1)


def test_low_ace_run_takes_the_five():
    assert can_place(table_with_low_ace_run(), Card(S, Rank.FIVE))


def test_low_ace_run_does_not_wrap_to_king():
    assert not can_place(table_with_low_ace_run(), Card(S, K))


def test_low_ace_run_rejects_a_second_ace():
    assert not can_place(table_with_low_ace_run(), Card(S, A))


def test_ace_extends_king_end_while_a_low_ace_run_is_also_on_the_table():
    # Round 3 [4, 4]: ♠A-2-3-4 and ♠10-J-Q-K.
    ctx = table_with(run(S, A, *ranks(2, 4)) + run(S, *ranks(10, 10), J, Q, K), round_idx=2)
    assert can_place(ctx, Card(S, A))


def test_set_accepts_any_card_of_its_rank():
    # Round 1 [3, 3]: 7s and 9s.
    ctx = table_with(
        [Card(S, Rank.SEVEN), Card(H, Rank.SEVEN), Card(C, Rank.SEVEN)] + FILLER_SET_OF_9S,
        round_idx=0,
    )
    assert can_place(ctx, Card(D, Rank.SEVEN))
    assert can_place(ctx, Card(S, Rank.SEVEN))  # duplicates are fine in a set
    assert not can_place(ctx, Card(D, Rank.EIGHT))


# ---------------------------------------------------------------------------
# Section 6: the player chooses where a Key Card goes
# ---------------------------------------------------------------------------

def table_with_7s_and_spade_run_3_to_6():
    # Round 2 [3, 4]: a set of 7s (no spade) and ♠3-4-5-6.
    sevens = [Card(H, Rank.SEVEN), Card(C, Rank.SEVEN), Card(D, Rank.SEVEN)]
    return table_with(run(S, *ranks(3, 6)) + sevens, round_idx=1)


def test_card_fitting_a_set_and_a_run_offers_both():
    ctx = table_with_7s_and_spade_run_3_to_6()
    assert offered_destinations(ctx, Card(S, Rank.SEVEN)) == {ON_SET, RUN_HIGH_END}


def test_placing_on_the_run_extends_the_run():
    ctx = table_with_7s_and_spade_run_3_to_6()
    place(ctx, Card(S, Rank.SEVEN), RUN_HIGH_END)
    assert can_place(ctx, Card(S, Rank.EIGHT))


def test_placing_on_the_set_leaves_the_run_unchanged():
    ctx = table_with_7s_and_spade_run_3_to_6()
    place(ctx, Card(S, Rank.SEVEN), ON_SET)
    assert not can_place(ctx, Card(S, Rank.EIGHT))
    assert can_place(ctx, Card(S, Rank.SEVEN))  # the run's high end is still open


def table_with_gap_between_runs():
    # Round 3 [4, 4]: ♠2-3-4-5 and ♠7-8-9-10, with a one-card gap at ♠6.
    return table_with(run(S, *ranks(2, 5)) + run(S, *ranks(7, 10)), round_idx=2)


def test_card_in_a_one_card_gap_offers_both_run_ends():
    ctx = table_with_gap_between_runs()
    assert offered_destinations(ctx, Card(S, Rank.SIX)) == {RUN_LOW_END, RUN_HIGH_END}


def test_gap_card_on_the_lower_run_opens_the_seven():
    ctx = table_with_gap_between_runs()
    place(ctx, Card(S, Rank.SIX), RUN_HIGH_END)  # ♠2-6
    assert can_place(ctx, Card(S, Rank.SEVEN))
    assert not can_place(ctx, Card(S, Rank.FIVE))


def test_gap_card_on_the_upper_run_opens_the_five():
    ctx = table_with_gap_between_runs()
    place(ctx, Card(S, Rank.SIX), RUN_LOW_END)  # ♠6-10
    assert can_place(ctx, Card(S, Rank.FIVE))
    assert not can_place(ctx, Card(S, Rank.SEVEN))


def test_ace_offers_whichever_run_end_it_fits():
    # ♠A-2-3-4 and ♠10-J-Q-K: the second Ace only fits above the King.
    ctx = table_with(run(S, A, *ranks(2, 4)) + run(S, *ranks(10, 10), J, Q, K), round_idx=2)
    assert offered_destinations(ctx, Card(S, A)) == {RUN_HIGH_END}


def test_placing_at_an_illegal_destination_is_refused():
    ctx = table_with_spade_run_3_to_6()
    with pytest.raises(ValueError):
        place(ctx, Card(S, Rank.SEVEN), RUN_LOW_END)


# ---------------------------------------------------------------------------
# Section 6: the player chooses which melds to go down with
# ---------------------------------------------------------------------------

def three_of(rank, suits=(S, H, D)):
    return [Card(suit, rank) for suit in suits]


def ctx_with_hand(cards, round_idx):
    ctx = GameContext(num_players=4)
    ctx.current_round_idx = round_idx
    player = ctx.players[0]
    player.hand_list.clear()
    player.private_hand.fill(0)
    player.receive_cards(cards)
    return ctx


SEVENS_NINES_QUEENS = three_of(Rank.SEVEN) + three_of(Rank.NINE) + three_of(Q)


def test_find_all_melds_lists_each_way_to_go_down_once():
    ctx = ctx_with_hand(SEVENS_NINES_QUEENS, round_idx=0)  # [3, 3]
    options = ctx.find_all_melds(ctx.players[0].private_hand, 2, 0)
    assert sorted(options) == sorted([
        [('set', Rank.SEVEN), ('set', Rank.NINE)],
        [('set', Rank.SEVEN), ('set', Q)],
        [('set', Rank.NINE), ('set', Q)],
    ])


def test_go_down_lays_the_chosen_sets():
    ctx = ctx_with_hand(SEVENS_NINES_QUEENS, round_idx=0)
    ctx.go_down(0, [('set', Rank.NINE), ('set', Q)])

    assert sorted(int(c.rank) for c in ctx.players[0].hand_list) == [Rank.SEVEN] * 3
    assert can_place(ctx, Card(C, Rank.NINE))
    assert not can_place(ctx, Card(C, Rank.SEVEN))


def test_go_down_lays_the_chosen_ace_high_run():
    # ♠A-2-3-4 or ♠J-Q-K-A from a single Ace; the player picks J-Q-K-A.
    hand = run(S, A, *ranks(2, 4), J, Q, K) + FILLER_SET_OF_9S
    ctx = ctx_with_hand(hand, round_idx=1)  # [3, 4]
    ctx.go_down(0, [('set', Rank.NINE), ('run', S, 10)])

    assert sorted(int(c.rank) for c in ctx.players[0].hand_list) == [Rank.TWO, Rank.THREE,
                                                                       Rank.FOUR]
    assert can_place(ctx, Card(S, Rank.TEN))
    assert not can_place(ctx, Card(S, Rank.FIVE))


def test_go_down_lets_a_set_and_run_share_a_rank_in_any_order():
    # ♠7 must go in the run; the set of 7s uses the other three.
    hand = run(S, *ranks(4, 7)) + three_of(Rank.SEVEN, (H, C, D))
    ctx = ctx_with_hand(hand, round_idx=1)
    ctx.go_down(0, [('set', Rank.SEVEN), ('run', S, 3)])
    assert ctx.players[0].hand_list == []


@pytest.mark.parametrize("melds", [
    pytest.param([('set', Rank.NINE)], id="too few melds"),
    pytest.param([('set', Rank.NINE), ('set', Q), ('set', Rank.SEVEN)], id="too many melds"),
    pytest.param([('set', Rank.NINE), ('set', K)], id="cards not held"),
    pytest.param([('set', Rank.NINE), ('run', S, 3)], id="wrong kind of meld"),
])
def test_go_down_refuses_an_invalid_choice_and_changes_nothing(melds):
    ctx = ctx_with_hand(SEVENS_NINES_QUEENS, round_idx=0)
    with pytest.raises(ValueError):
        ctx.go_down(0, melds)
    assert len(ctx.players[0].hand_list) == 9
    assert not ctx.players[0].is_down
    assert ctx.table_sets.sum() == 0 and ctx.runs_on_table == []


def reach_go_down_decision(engine_cls):
    """First turn of round 1 [3, 3], with the active player holding 7s, 9s and Queens."""
    ctx, engine = start_first_turn(engine_cls)
    player = ctx.active_player
    player.hand_list.clear()
    player.private_hand.fill(0)
    player.receive_cards(SEVENS_NINES_QUEENS + [Card(C, Rank.TWO), Card(C, Rank.FIVE)])
    engine.resolve_pickup(0)
    may_i_offers(ctx, engine)
    assert engine.state_id == "go_down_decision"
    return ctx, engine


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_engine_goes_down_with_the_chosen_melds(engine_cls):
    ctx, engine = reach_go_down_decision(engine_cls)
    engine.resolve_go_down(GO_DOWN, [('set', Rank.SEVEN), ('set', Q)])

    assert engine.state_id == "table_play_phase"
    assert ctx.active_player.is_down
    assert ctx.table_sets[:, Rank.SEVEN].sum() == 3
    assert ctx.table_sets[:, Q].sum() == 3
    assert ctx.table_sets[:, Rank.NINE].sum() == 0


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_engine_refuses_an_invalid_go_down_choice(engine_cls):
    ctx, engine = reach_go_down_decision(engine_cls)
    with pytest.raises(ValueError):
        engine.resolve_go_down(GO_DOWN, [('set', K), ('set', Q)])
    assert not ctx.active_player.is_down


# ---------------------------------------------------------------------------
# auto_place_cards (used by ProbabilisticAgent's go-down look-ahead)
# ---------------------------------------------------------------------------

def table_with_spade_run_9_to_q():
    return table_with(run(S, *ranks(9, 9), Rank.TEN, J, Q) + FILLER_SET_OF_9S, round_idx=1)


def test_auto_place_cards_conserves_cards():
    ctx = table_with_spade_run_9_to_q()
    ctx.players[0].receive_cards([Card(S, K)])
    before = verify_mass_integrity(ctx=ctx, test_case=_Asserter())

    ctx.auto_place_cards(0)

    assert ctx.players[0].hand_list == []
    assert verify_mass_integrity(ctx=ctx, test_case=_Asserter()) == before


def test_auto_place_cards_places_cards_unlocked_by_earlier_placements():
    # ♠8 only fits after ♠7 is placed; hand order puts ♠8 first.
    ctx = table_with_spade_run_3_to_6()
    ctx.players[0].receive_cards([Card(S, Rank.EIGHT), Card(S, Rank.SEVEN)])

    ctx.auto_place_cards(0)

    assert ctx.players[0].hand_list == []


class _Asserter:
    """Minimal stand-in for unittest.TestCase so verify_mass_integrity works in pytest."""

    def assertEqual(self, first, second, msg=None):
        assert first == second, msg


# ---------------------------------------------------------------------------
# Section 2: Setup & the deal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine_cls", ENGINES)
def test_player_after_dealer_takes_first_turn(engine_cls):
    ctx = GameContext(num_players=4)
    engine = engine_cls(ctx)
    engine.start_game()
    engine.deal_cards()
    assert ctx.active_player_idx == (ctx.dealer_idx + 1) % 4


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_first_dealer_is_random(engine_cls):
    first_dealers = set()
    for seed in range(40):
        random.seed(seed)
        ctx = GameContext(num_players=4)
        engine_cls(ctx).start_game()
        first_dealers.add(ctx.dealer_idx)
    assert first_dealers == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Section 5: May-I
# ---------------------------------------------------------------------------

def start_first_turn(engine_cls, num_players=4, seed=1234):
    random.seed(seed)
    ctx = GameContext(num_players=num_players)
    engine = engine_cls(ctx)
    engine.start_game()
    engine.deal_cards()
    engine.enter_pickup()
    return ctx, engine


def may_i_offers(ctx, engine):
    """Everyone passes; returns the order in which players were offered the May-I."""
    offered = []
    while engine.state_id == "may_i_decision":
        offered.append(ctx.may_i_target_idx)
        engine.resolve_may_i(3)  # PASS
    return offered


@pytest.mark.parametrize("num_players", [3, 4])
@pytest.mark.parametrize("engine_cls", ENGINES)
def test_opening_upcard_is_offered_to_everyone_but_the_active_player(engine_cls, num_players):
    ctx, engine = start_first_turn(engine_cls, num_players)
    active = ctx.active_player_idx

    engine.resolve_pickup(0)  # PICK_STOCK

    expected = [(active + i) % num_players for i in range(1, num_players)]
    assert may_i_offers(ctx, engine) == expected


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_may_i_is_not_offered_to_the_discarder(engine_cls):
    ctx, engine = start_first_turn(engine_cls)
    # Play out the first turn so the next top discard has a known discarder.
    first = ctx.active_player_idx
    engine.resolve_pickup(0)
    may_i_offers(ctx, engine)
    while engine.state_id != "discard_phase":
        engine.resolve_go_down(5)  # WAIT
    engine.perform_discard(0)
    engine.enter_pickup()
    assert ctx.active_player_idx == (first + 1) % 4

    engine.resolve_pickup(0)

    offered = may_i_offers(ctx, engine)
    assert first not in offered
    assert ctx.active_player_idx not in offered


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_may_i_is_not_offered_to_players_who_are_down_or_out_of_may_is(engine_cls):
    ctx, engine = start_first_turn(engine_cls)
    active = ctx.active_player_idx
    down, used_up = (active + 1) % 4, (active + 2) % 4
    ctx.players[down].is_down = True
    ctx.players[used_up].may_is_used = 3

    engine.resolve_pickup(0)

    offered = may_i_offers(ctx, engine)
    assert down not in offered
    assert used_up not in offered


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_may_i_caller_takes_discard_plus_one_stock_card(engine_cls):
    ctx, engine = start_first_turn(engine_cls)
    caller = ctx.players[(ctx.active_player_idx + 1) % 4]
    target = ctx.discard_pile[-1]
    hand_before = list(caller.hand_list)

    engine.resolve_pickup(0)
    engine.resolve_may_i(2)  # CALL_MAY_I

    new_cards = [c for c in caller.hand_list if not any(c is b for b in hand_before)]
    assert len(new_cards) == 2
    assert any(c is target for c in new_cards)
    assert caller.may_is_used == 1


# ---------------------------------------------------------------------------
# Section 7: Stock exhaustion
# ---------------------------------------------------------------------------

def rig_stock_and_discards(ctx, stock_card, discards):
    """Replaces the stock and discard pile; other players are made ineligible to May-I."""
    ctx.deck = [stock_card]
    ctx.discard_pile = list(discards)
    ctx.dead_cards.fill(0)
    for card in discards[:-1]:
        ctx._sync_ace(ctx.dead_cards, int(card.suit), int(card.rank), increment=True)


def fresh_cards(n):
    """Distinct card objects to track by identity."""
    return [Card(S, Rank(i), deck_index=1) for i in range(n)]


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_stock_is_rebuilt_straight_after_the_draw_that_empties_it(engine_cls):
    ctx, engine = start_first_turn(engine_cls)
    last_stock, a, b, top = fresh_cards(4)
    rig_stock_and_discards(ctx, last_stock, [a, b, top])
    for i, p in enumerate(ctx.players):
        if i != ctx.active_player_idx:
            p.is_down = True  # nobody can May-I

    engine.resolve_pickup(0)

    assert any(c is last_stock for c in ctx.active_player.hand_list)
    assert ctx.discard_pile == [top]
    assert sorted(c.id for c in ctx.deck) == sorted([a.id, b.id])


@pytest.mark.parametrize("engine_cls", ENGINES)
def test_may_i_caller_takes_the_last_stock_card(engine_cls):
    """The worked example in rules.md Section 7."""
    ctx, engine = start_first_turn(engine_cls)
    last_stock, a, b, c, target = fresh_cards(5)
    rig_stock_and_discards(ctx, last_stock, [a, b, c, target])
    active = ctx.active_player
    caller = ctx.players[(ctx.active_player_idx + 1) % 4]

    engine.resolve_pickup(0)
    engine.resolve_may_i(2)

    assert any(x is target for x in caller.hand_list)
    assert any(x is last_stock for x in caller.hand_list)
    # Stock rebuilt from [a, b]; c stays as the discard. The active player draws a or b.
    assert ctx.discard_pile == [c]
    assert any(x is a or x is b for x in active.hand_list)
    assert len(ctx.deck) == 1
    # c was exposed by the May-I, so it stays buried until the active player covers it.
    assert ctx.dead_cards[:, :13].sum() == 1
    assert ctx.dead_cards[int(c.suit), int(c.rank)] == 1
