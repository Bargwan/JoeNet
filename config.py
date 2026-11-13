from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class JoeConfig:
    """
    Holds penalty point values and global game settings.
    Injected into GameContext to decouple scoring math from core logic.
    """
    points_ace: int = 20
    points_eight_to_king: int = 10
    points_two_to_seven: int = 5

    # Engine safety net (not a rule): a round is ended and scored normally after this many
    # full circuits of the table. ~15 circuits of 4 players exhaust the 59-card stock once.
    max_turns: int = 100

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