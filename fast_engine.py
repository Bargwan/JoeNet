import logging

logger = logging.getLogger(__name__)


class FastState:
    """Lightweight state representation to mimic statemachine properties."""
    __slots__ = ['id']

    def __init__(self, state_id):
        self.id = state_id

    def __eq__(self, other):
        # Allows for fast matching against other states or raw strings
        if isinstance(other, FastState):
            return self.id == other.id
        elif isinstance(other, str):
            return self.id == other
        return False

    def __repr__(self):
        return f"State({self.id})"

class JoeEngine:
    """
    A lightning-fast, hardcoded state machine replacement for the 'Joe' card game.
    """
    __slots__ = ['ctx', 'current_state']

    # Refactored to match original engine.py Title Case strings
    setup = FastState('Setup')
    dealing = FastState('Dealing')
    start_turn = FastState('Start Turn')
    pickup_decision = FastState('Pickup Decision')
    processing_pickup = FastState('Processing Pickup')
    may_i_check = FastState('May-I Check')
    may_i_decision = FastState('May-I Decision')
    go_down_decision = FastState('Go Down Decision')
    going_down = FastState('Going Down')
    table_play_phase = FastState('Table Play Phase')
    processing_table_play = FastState('Processing Table Play')
    discard_phase = FastState('Discard Phase')
    processing_discard = FastState('Processing Discard')
    round_end = FastState('Round End')
    game_over = FastState('Game Over')

    def __init__(self, context):
        self.ctx = context
        self.current_state = self.setup

    @property
    def state_id(self) -> str:
        """
        Universal normalizer for the fast engine. Guarantees it reports
        its current state as a standard snake_case string, perfectly
        mirroring the behavior of the python-statemachine engine.
        """
        return self.current_state.id.lower().replace(' ', '_').replace('-', '_')

    # ========================================================================
    # 1. TRANSITIONS (Replaces python-statemachine dispatching)
    # ========================================================================

    def start_game(self):
        self.current_state = self.dealing
        self.on_enter_dealing()

    def deal_cards(self):
        self.current_state = self.start_turn
        self.on_enter_start_turn()

    def enter_pickup(self):
        self.current_state = self.pickup_decision

    def resolve_pickup(self, action: int):
        self.before_resolve_pickup(action)
        if action == 1:  # PICK_DISCARD
            self.current_state = self.processing_pickup
            self.on_enter_processing_pickup()
        elif action == 0:  # PICK_STOCK
            self.current_state = self.may_i_check
            self.on_enter_may_i_check()

    def evaluate_may_i_target(self):
        if self.ctx.all_may_i_targets_checked():
            self.current_state = self.processing_pickup
            self.on_enter_processing_pickup()
        elif self.ctx.is_may_i_target_eligible():
            self.current_state = self.may_i_decision

    def skip_ineligible_target(self):
        self.before_skip_ineligible_target()
        self.current_state = self.may_i_check
        self.on_enter_may_i_check()

    def resolve_may_i(self, action: int):
        self.before_resolve_may_i(action)
        if action == 3:  # PASS
            self.current_state = self.may_i_check
            self.on_enter_may_i_check()
        elif action == 2:  # CALL_MAY_I
            self.current_state = self.processing_pickup
            self.on_enter_processing_pickup()

    def evaluate_hand(self):
        if self.ctx.active_player.is_down:
            self.current_state = self.table_play_phase
        elif not self.ctx.active_player.is_down and not self.ctx.check_hand_objective(
                self.ctx.active_player_idx):
            self.current_state = self.discard_phase
        elif not self.ctx.active_player.is_down and self.ctx.check_hand_objective(
                self.ctx.active_player_idx):
            self.current_state = self.go_down_decision

    def resolve_go_down(self, action: int):
        if action == 4:  # GO_DOWN
            self.current_state = self.going_down
            self.on_enter_going_down()
        elif action == 5:  # WAIT
            self.current_state = self.discard_phase

    def commit_melds(self):
        self.on_commit_melds()
        self.current_state = self.table_play_phase

    def perform_table_play(self, card_index):
        self.current_state = self.processing_table_play
        self.on_enter_processing_table_play(card_index)

    def evaluate_table_play(self):
        self.current_state = self.table_play_phase

    def end_table_play(self):
        if len(self.ctx.active_player.hand_list) == 0:
            self.current_state = self.round_end
            self.on_enter_round_end()
        else:
            self.current_state = self.discard_phase

    def perform_discard(self, card_index):
        self.current_state = self.processing_discard
        self.on_enter_processing_discard(card_index)

    def evaluate_discard_result(self):
        if len(self.ctx.active_player.hand_list) == 0:
            self.current_state = self.round_end
            self.on_enter_round_end()
        elif self.ctx.total_actions >= self.ctx.config.max_actions:
            self.current_state = self.round_end
            self.on_enter_round_end()
        elif self.ctx.current_circuit >= self.ctx.config.max_turns:
            self.current_state = self.round_end
            self.on_enter_round_end()
        else:
            self.current_state = self.start_turn
            self.on_enter_start_turn()

    def resolve_round_end(self):
        if self.ctx.current_round_idx >= 7:
            self.current_state = self.game_over
        else:
            self.current_state = self.dealing
            self.on_enter_dealing()

    # ========================================================================
    # 2. CALLBACKS (Action Hooks)
    # ========================================================================

    def on_enter_dealing(self):
        if self.ctx.current_round_idx > 0:
            self.ctx.rotate_dealer()
        self.ctx.execute_deal()

    def on_enter_start_turn(self):
        self.ctx.rotate_player()
        self.ctx.advance_action_counter()

        # NOTE: You may want to comment this debug line out during Phase 1 Data Generation
        # to save string-formatting micro-seconds across the millions of turns.
        logger.debug(
            f"--- START TURN: Player {self.ctx.active_player_idx} (Circuit {self.ctx.current_circuit}, Action {self.ctx.total_actions}) ---")

    def before_resolve_pickup(self, action: int):
        if action == 1:  # PICK_DISCARD
            self.ctx.execute_pickup_discard()
        elif action == 0:  # PICK_STOCK
            self.ctx.execute_pickup_stock()
            self.ctx.start_may_i_checks()

    def on_enter_processing_pickup(self):
        self.evaluate_hand()

    def on_enter_may_i_check(self):
        if self.ctx.all_may_i_targets_checked() or self.ctx.is_may_i_target_eligible():
            self.evaluate_may_i_target()
        else:
            self.skip_ineligible_target()

    def before_skip_ineligible_target(self):
        self.ctx.advance_may_i_target()

    def before_resolve_may_i(self, action: int):
        if action == 3:  # PASS
            self.ctx.advance_may_i_target()
        elif action == 2:  # CALL_MAY_I
            self.ctx.execute_may_i_call()

    def on_enter_going_down(self):
        self.ctx.active_player.is_down = True
        self.commit_melds()

    def on_commit_melds(self):
        self.ctx.go_down(self.ctx.active_player_idx)

    def on_enter_processing_table_play(self, card_index):
        self.ctx.execute_table_play(self.ctx.active_player_idx, card_index)
        self.evaluate_table_play()

    def on_enter_processing_discard(self, card_index):
        self.ctx.execute_discard(self.ctx.active_player_idx, card_index)
        self.evaluate_discard_result()

    def on_enter_round_end(self):
        self.ctx.calculate_scores()
        self.resolve_round_end()