import logging
from statemachine import State, StateMachine

logger = logging.getLogger(__name__)


class JoeEngine(StateMachine):
    """
    State machine for the 'Joe' card game.
    Organized chronologically by Game Phase.
    """

    def __init__(self, context):
        self.ctx = context
        super().__init__()

    @property
    def state_id(self) -> str:
        """
        Universal normalizer. Guarantees the engine always reports its
        current state as a standard snake_case string, regardless of
        the underlying python-statemachine version or human-readable name.
        """
        return self.current_state.name.lower().replace(' ', '_').replace('-', '_')

    # ========================================================================
    # 1. STATES
    # ========================================================================

    # --- Phase: Initialization ---
    setup = State('Setup', initial=True)
    dealing = State('Dealing')

    # --- Phase: Turn Start & Pickup ---
    start_turn = State('Start Turn')
    pickup_decision = State('Pickup Decision')  # WAITING STATE
    processing_pickup = State('Processing Pickup')  # AUTOMATIC

    # --- Phase: May-I Sub-Loop ---
    may_i_check = State('May-I Check')  # AUTOMATIC
    may_i_decision = State('May-I Decision')  # WAITING STATE

    # --- Phase: Hand Evaluation & Going Down ---
    go_down_decision = State('Go Down Decision')  # WAITING STATE
    going_down = State('Going Down')  # AUTOMATIC

    # --- Phase: Table Play ---
    table_play_phase = State('Table Play Phase')  # WAITING STATE
    processing_table_play = State('Processing Table Play')  # AUTOMATIC

    # --- Phase: Discard ---
    discard_phase = State('Discard Phase')  # WAITING STATE
    processing_discard = State('Processing Discard')  # AUTOMATIC

    # --- Phase: Resolution ---
    round_end = State('Round End')  # AUTOMATIC
    game_over = State('Game Over', final=True)

    # ========================================================================
    # 2. TRANSITIONS
    # ========================================================================

    # --- Phase: Initialization ---
    start_game = setup.to(dealing)
    deal_cards = dealing.to(start_turn)

    # --- Phase: Turn Start & Pickup ---
    enter_pickup = start_turn.to(pickup_decision)
    resolve_pickup = (
            pickup_decision.to(processing_pickup, cond="action_is_discard")
            | pickup_decision.to(may_i_check, cond="action_is_stock")
    )

    # --- Phase: May-I Sub-Loop ---
    evaluate_may_i_target = (
            may_i_check.to(processing_pickup, cond="all_targets_checked")
            | may_i_check.to(may_i_decision, cond="target_is_eligible")
    )
    skip_ineligible_target = may_i_check.to(may_i_check, cond="target_is_ineligible")
    resolve_may_i = (
            may_i_decision.to(processing_pickup, cond="action_is_call")
            | may_i_decision.to(may_i_check, cond="action_is_pass")
    )

    # --- Phase: Hand Evaluation & Going Down ---
    evaluate_hand = (
            processing_pickup.to(table_play_phase, cond="is_already_down")
            | processing_pickup.to(discard_phase, cond="cannot_go_down")
            | processing_pickup.to(go_down_decision, cond="can_go_down_valid")
    )
    resolve_go_down = (
            go_down_decision.to(going_down, cond="action_is_go_down")
            | go_down_decision.to(discard_phase, cond="action_is_wait")
    )
    commit_melds = going_down.to(table_play_phase)

    # --- Phase: Table Play ---
    perform_table_play = table_play_phase.to(processing_table_play)
    evaluate_table_play = processing_table_play.to(table_play_phase)
    end_table_play = (
            table_play_phase.to(round_end, cond="hand_is_empty")
            | table_play_phase.to(discard_phase, cond="hand_not_empty")
    )

    # --- Phase: Discard ---
    perform_discard = discard_phase.to(processing_discard)
    evaluate_discard_result = (
            processing_discard.to(round_end, cond="hand_is_empty")
            | processing_discard.to(round_end, cond="max_turns_reached")
            | processing_discard.to(start_turn, cond="hand_not_empty")
    )

    # --- Phase: Resolution ---
    resolve_round_end = (
            round_end.to(game_over, cond="is_tournament_over")
            | round_end.to(dealing)
    )

# ========================================================================
    # 3. GUARDS (Conditionals)
    # ========================================================================

    def action_is_stock(self, action: int):
        return action == 0  # PICK_STOCK

    def action_is_discard(self, action: int):
        return action == 1  # PICK_DISCARD

    def action_is_call(self, action: int):
        return action == 2  # CALL_MAY_I

    def action_is_pass(self, action: int):
        return action == 3  # PASS

    def action_is_go_down(self, action: int):
        return action == 4  # GO_DOWN

    def action_is_wait(self, action: int):
        return action == 5  # WAIT

    def is_already_down(self):
        return self.ctx.active_player.is_down

    def cannot_go_down(self):
        return not self.ctx.active_player.is_down and not self.ctx.check_hand_objective(
            self.ctx.active_player_idx)

    def can_go_down_valid(self):
        return not self.ctx.active_player.is_down and self.ctx.check_hand_objective(
            self.ctx.active_player_idx)

    def all_targets_checked(self):
        return self.ctx.all_may_i_targets_checked()

    def target_is_eligible(self):
        return self.ctx.is_may_i_target_eligible()

    def target_is_ineligible(self):
        return not self.ctx.is_may_i_target_eligible()

    def hand_is_empty(self):
        return len(self.ctx.active_player.hand_list) == 0

    def hand_not_empty(self):
        return not self.hand_is_empty()

    def is_tournament_over(self):
        return self.ctx.current_round_idx >= 7

    def max_turns_reached(self):
        return self.ctx.total_actions >= self.ctx.config.max_actions

    # ========================================================================
    # 4. CALLBACKS (Action Hooks)
    # ========================================================================

    def on_enter_dealing(self):
        if self.ctx.current_round_idx > 0:
            self.ctx.rotate_dealer()
        self.ctx.execute_deal()

    def on_enter_start_turn(self):
        self.ctx.rotate_player()
        self.ctx.advance_action_counter()
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
        if self.all_targets_checked() or self.target_is_eligible():
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

    def on_enter_processing_table_play(self, card_index: int):
        self.ctx.execute_table_play(self.ctx.active_player_idx, card_index)
        self.evaluate_table_play()

    def on_enter_processing_discard(self, card_index: int):
        self.ctx.execute_discard(self.ctx.active_player_idx, card_index)
        self.evaluate_discard_result()

    def on_enter_round_end(self):
        self.ctx.calculate_scores()
        self.resolve_round_end()


def generate_diagram():
    from statemachine.contrib.diagram import DotGraphMachine
    graph = DotGraphMachine(JoeEngine)
    dot = graph()
    dot.set_rankdir("TB")
    dot.set_dpi(300)
    dot.write_png("joe_engine_flow.png")
    print("Graph successfully generated and saved as 'joe_engine_flow.png'")


if __name__ == '__main__':
    generate_diagram()