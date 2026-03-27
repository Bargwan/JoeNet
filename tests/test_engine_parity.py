import unittest

from game_context import GameContext
from config import JoeConfig

# Import both engines to prove they behave identically
from fast_engine import JoeEngine as FastEngine
from engine import JoeEngine as SlowEngine


class TestEngineParityAndBugs(unittest.TestCase):

    def get_state_id(self, engine):
        """
        Helper to normalize the state ID across both engine types.
        python-statemachine and FastState format IDs slightly differently.
        """
        # Handle python-statemachine's new API (SlowEngine)
        if hasattr(engine, 'configuration'):
            # 'configuration' returns a list of currently active states
            raw_id = engine.configuration[0].id
        else:
            # Our custom FastEngine
            raw_id = engine.current_state.id

        return str(raw_id).lower().replace(' ', '_')

    def test_auto_stepper_progression(self):
        """
        Verify Bug #2: The engine requires explicit auto-stepper calls
        to navigate non-decision initialization states safely.
        """
        for EngineClass in [FastEngine, SlowEngine]:
            with self.subTest(engine=EngineClass.__module__):
                ctx = GameContext(num_players=4, config=JoeConfig())
                engine = EngineClass(ctx)

                self.assertEqual(self.get_state_id(engine), 'setup')

                # Step 1: Start Game
                engine.start_game()
                self.assertEqual(self.get_state_id(engine), 'dealing')

                # Step 2: Deal Cards
                engine.deal_cards()
                self.assertEqual(self.get_state_id(engine), 'start_turn')

                # Step 3: Enter Pickup
                engine.enter_pickup()
                self.assertEqual(self.get_state_id(engine), 'pickup_decision')

                # At this point, the game is paused waiting for Neural Network input.
                # Because the dealer was 0, Player 1 should be the active player.
                self.assertEqual(ctx.active_player_idx, 1,
                                 "Player 1 should be active after the first rotation")

    def test_may_i_target_trap(self):
        """
        Verify Bug #1: The May-I decision must be evaluated for the target player,
        not the active player, and the target correctly receives the penalty cards.
        """
        for EngineClass in [FastEngine, SlowEngine]:
            with self.subTest(engine=EngineClass.__module__):
                ctx = GameContext(num_players=4, config=JoeConfig())
                engine = EngineClass(ctx)

                # Auto-step to the first decision
                engine.start_game()
                engine.deal_cards()
                engine.enter_pickup()

                active_player = ctx.active_player_idx
                self.assertEqual(active_player, 1)

                # Ensure discard pile exists and Player 2 hand size is standard 11
                self.assertEqual(len(ctx.discard_pile), 1)
                self.assertEqual(len(ctx.players[2].hand_list), 11)

                # ACT: Active Player (1) picks from the STOCK (Action 0)
                # This leaves the top discard available, triggering the May-I queue
                engine.resolve_pickup(0)

                # ASSERT: Engine halts at May-I Decision
                self.assertEqual(self.get_state_id(engine), 'may_i_decision')

                # BUG #1 CHECK: Target is Player 2 (Downstream of Player 1)
                self.assertEqual(ctx.may_i_target_idx, 2,
                                 "Target must be downstream of active player")
                self.assertNotEqual(ctx.may_i_target_idx, ctx.active_player_idx,
                                    "Target CANNOT be the active player")

                # ACT: Player 2 calls the May-I (Action 2)
                engine.resolve_may_i(2)

                # ASSERT: Data movement
                self.assertEqual(len(ctx.discard_pile), 0,
                                 "Discard pile should be empty after May-I")
                self.assertEqual(len(ctx.players[2].hand_list), 13,
                                 "Player 2 should have received 1 discard + 1 penalty stock card")
                self.assertEqual(ctx.players[2].may_is_used, 1,
                                 "Player 2 May-I counter should increment")

                # ASSERT: Engine resumes for Active Player (1)
                # The engine will auto-route through `processing_pickup` -> `evaluate_hand()`
                # Because they likely don't have a valid meld yet, it safely lands on 'discard_phase'
                current_state = self.get_state_id(engine)
                self.assertIn(current_state,
                              ['discard_phase', 'go_down_decision', 'table_play_phase'])
                self.assertEqual(ctx.active_player_idx, 1,
                                 "Control must return to the active player after May-I completes!")


if __name__ == '__main__':
    unittest.main()