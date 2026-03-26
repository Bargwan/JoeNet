import numpy as np


def verify_mass_integrity(test_case, ctx, expected_total=None):
    """
    Shared helper to mathematically prove that no cards have been created,
    destroyed, or desynchronized between Python lists and NumPy tensors.
    """
    total_cards = 0

    # 1. Deck
    total_cards += len(ctx.deck)

    # 2. Discard Pile & Dead Cards Sync
    total_cards += len(ctx.discard_pile)

    # dead_cards tensor should exactly equal discard list minus the top card
    expected_dead = len(ctx.discard_pile) - 1 if ctx.discard_pile else 0
    actual_dead = np.sum(ctx.dead_cards[:, 0:13])
    test_case.assertEqual(
        expected_dead, actual_dead,
        f"Discard Sync Error: List has {len(ctx.discard_pile)}, dead_cards has {actual_dead}."
    )

    # 3. Player Hands Sync
    for i, player in enumerate(ctx.players):
        list_count = len(player.hand_list)
        tensor_count = np.sum(player.private_hand[:, 0:13])

        test_case.assertEqual(
            list_count, tensor_count,
            f"Player {i} Sync Error: List={list_count}, Tensor={tensor_count}."
        )
        total_cards += list_count

    # 4. Table Tensors
    table_set_count = np.sum(ctx.table_sets[:, 0:13])
    table_run_count = np.sum(ctx.table_runs[:, 0:13])
    total_cards += (table_set_count + table_run_count)

    # 5. Global Total Assertion
    if expected_total is not None:
        test_case.assertEqual(
            total_cards, expected_total,
            f"Mass Conservation Error: Expected {expected_total}, found {total_cards}."
        )

    return total_cards