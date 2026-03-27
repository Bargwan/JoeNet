from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class JoeConfig:
    """
    Holds unscaled terminal point values, global game settings,
    and Reinforcement Learning hyperparameters.
    Injected into GameContext to decouple scoring math from core logic.
    """
    points_ace: int = 20
    points_eight_to_king: int = 10
    points_two_to_seven: int = 5

    # A "turn" is defined as one complete circuit of the board (all players).
    # 15 circuits will roughly exhaust the 60-card Stock Pile once.
    max_turns: int = 30

    # An "action" is an individual decision point (Pickup, May-I, Discard, etc.)
    # Used strictly as an absolute engine failsafe against infinite RL loops.
    max_actions: int = 500

    # RL Hyperparameters for Asymmetric Terminal Scoring
    catch_up_multiplier: float = 2.0
    pull_ahead_multiplier: float = 0.5

    # Round Index -> (Sets Needed, Runs Needed)
    objective_map: Dict[int, Tuple[int, int]] = field(default_factory=lambda: {
        0: (2, 0),  # [3,3]
        1: (1, 1),  # [3,4]
        2: (0, 2),  # [4,4]
        3: (3, 0),  # [3,3,3]
        4: (2, 1),  # [3,3,4]
        5: (1, 2),  # [3,4,4]
        6: (0, 3),  # [4,4,4]
    })