import unittest

from game_context import GameContext
from config import JoeConfig

# Import both engines to prove they behave identically
from fast_engine import JoeEngine as FastEngine
from engine import JoeEngine as SlowEngine


class TestEngineParityAndBugs(unittest.TestCase):

    def get_state_id(self, engine):
        """The current state as a snake_case string, for either engine."""
        # python-statemachine's API (SlowEngine)
        if hasattr(engine, 'configuration'):
            raw_id = engine.configuration[0].id
        else:
            # Our custom FastEngine
            raw_id = engine.current_state.id

        return str(raw_id).lower().replace(' ', '_').replace('-', '_')

    def test_auto_stepper_progression(self):
        """
        Verify both engines need explicit calls (start_game, deal_cards,
        enter_pickup) to step through the setup states to the first decision.
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

                # The player clockwise of the (random) dealer takes the first turn.
                self.assertEqual(ctx.active_player_idx, (ctx.dealer_idx + 1) % 4,
                                 "The player after the dealer should be active first")

    def test_may_i_target_trap(self):
        """
        Verify the May-I decision belongs to the target player, not the active
        player, and that the target receives the discard plus the penalty card.
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
                target = (active_player + 1) % 4

                # Ensure discard pile exists and the target's hand size is standard 11
                self.assertEqual(len(ctx.discard_pile), 1)
                self.assertEqual(len(ctx.players[target].hand_list), 11)

                # ACT: Active Player (1) picks from the STOCK (Action 0)
                # This leaves the top discard available, triggering the May-I queue
                engine.resolve_pickup(0)

                # ASSERT: Engine halts at May-I Decision
                self.assertEqual(self.get_state_id(engine), 'may_i_decision')

                # BUG #1 CHECK: Target is the player downstream of the active player
                self.assertEqual(ctx.may_i_target_idx, target,
                                 "Target must be downstream of active player")
                self.assertNotEqual(ctx.may_i_target_idx, ctx.active_player_idx,
                                    "Target CANNOT be the active player")

                # ACT: The target calls the May-I (action 2)
                engine.resolve_may_i(2)

                # ASSERT: Data movement
                self.assertEqual(len(ctx.discard_pile), 0,
                                 "Discard pile should be empty after May-I")
                self.assertEqual(len(ctx.players[target].hand_list), 13,
                                 "Target should have received 1 discard + 1 penalty stock card")
                self.assertEqual(ctx.players[target].may_is_used, 1,
                                 "Target May-I counter should increment")

                # ASSERT: Engine resumes for Active Player (1)
                # The engine will auto-route through `processing_pickup` -> `evaluate_hand()`
                # Because they likely don't have a valid meld yet, it safely lands on 'discard_phase'
                current_state = self.get_state_id(engine)
                self.assertIn(current_state,
                              ['discard_phase', 'go_down_decision', 'table_play_phase'])
                self.assertEqual(ctx.active_player_idx, active_player,
                                 "Control must return to the active player after May-I completes!")


    def test_full_games_match_between_engines(self):
        """Both engines play identical full games given the same seed and agents."""
        import random

        from agents import HeuristicAgent
        from evaluate_arena import apply_engine_action

        def play(EngineClass, seed):
            random.seed(seed)
            ctx = GameContext(num_players=4, config=JoeConfig())
            engine = EngineClass(ctx)
            agents = [HeuristicAgent(random_seed=i) for i in range(4)]
            engine.start_game()
            while engine.state_id != 'game_over':
                state = engine.state_id
                if state == 'dealing':
                    engine.deal_cards()
                elif state == 'start_turn':
                    engine.enter_pickup()
                else:
                    idx = ctx.may_i_target_idx if state == 'may_i_decision' else ctx.active_player_idx
                    action = agents[idx].select_action(state, ctx, idx,
                                                       ctx.get_action_mask(idx, state))
                    apply_engine_action(engine, ctx, state, idx, action, agents[idx])
            return [p.score for p in ctx.players]

        self.assertEqual(play(FastEngine, 0), play(SlowEngine, 0))


if __name__ == '__main__':
    unittest.main()