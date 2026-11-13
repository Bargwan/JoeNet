"""
Drives either engine with action objects (see actions.py).

Both engine.JoeEngine and fast_engine.JoeEngine inherit EngineDriver, so the
game is played the same way on either:

    engine.advance()                      # deal and start the first turn
    while (decision := engine.decider()):
        state_id, player_idx = decision
        legal = ctx.legal_actions(player_idx, state_id)
        engine.apply(chosen_action)       # plays it and moves on to the next decision
"""
from actions import (
    CallMayI,
    Discard,
    EndTablePlay,
    GoDown,
    Pass,
    PickDiscard,
    PickStock,
    Place,
    Wait,
)

# Engine action numbers for the transitions that still take them
_PICK_STOCK, _PICK_DISCARD, _CALL_MAY_I, _PASS, _GO_DOWN, _WAIT = range(6)

DECISION_STATES = ('pickup_decision', 'may_i_decision', 'go_down_decision',
                   'table_play_phase', 'discard_phase')


class EngineDriver:
    """Mixin for the engines: whose decision it is, and applying action objects."""
    __slots__ = ()

    def decider(self):
        """
        (state_id, player_idx) for the decision the game is waiting on, or None
        when it isn't waiting on one (the game is over, or advance() is needed).
        The May-I decision belongs to the player being offered it.
        """
        state_id = self.state_id
        if state_id not in DECISION_STATES:
            return None
        if state_id == 'may_i_decision':
            return state_id, self.ctx.may_i_target_idx
        return state_id, self.ctx.active_player_idx

    def advance(self):
        """Steps through the states that need no decision: setup, dealing and turn start."""
        while True:
            state_id = self.state_id
            if state_id == 'setup':
                self.start_game()
            elif state_id == 'dealing':
                self.deal_cards()
            elif state_id == 'start_turn':
                self.enter_pickup()
            else:
                return

    def apply(self, action):
        """
        Plays an action for the player whose decision it is, then advances to the
        next decision. Raises ValueError, changing nothing, if the action isn't legal.
        """
        decision = self.decider()
        if decision is None:
            raise ValueError(f"The game is not waiting on a decision (state: {self.state_id}).")
        state_id, player_idx = decision
        if action not in self.ctx.legal_actions(player_idx, state_id):
            raise ValueError(f"{action} is not legal in {state_id} for player {player_idx}.")

        if isinstance(action, PickStock):
            self.resolve_pickup(_PICK_STOCK)
        elif isinstance(action, PickDiscard):
            self.resolve_pickup(_PICK_DISCARD)
        elif isinstance(action, CallMayI):
            self.resolve_may_i(_CALL_MAY_I)
        elif isinstance(action, Pass):
            self.resolve_may_i(_PASS)
        elif isinstance(action, GoDown):
            self.resolve_go_down(_GO_DOWN, list(action.melds))
        elif isinstance(action, Wait):
            self.resolve_go_down(_WAIT)
        elif isinstance(action, Place):
            self.perform_table_play(self._hand_index(player_idx, action), action.destination)
        elif isinstance(action, EndTablePlay):
            self.end_table_play()
        elif isinstance(action, Discard):
            self.perform_discard(self._hand_index(player_idx, action))
        self.advance()

    def _hand_index(self, player_idx, action):
        """Index in the player's hand of a card with the action's suit and rank."""
        for i, card in enumerate(self.ctx.players[player_idx].hand_list):
            if int(card.suit) == action.suit and int(card.rank) == action.rank:
                return i
        raise ValueError(f"Player {player_idx} does not hold the card for {action}.")
