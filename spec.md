# Specification: JoeNet

## 1. Overview
JoeNet builds computer players for **Joe**, a multiplayer imperfect-information Rummy variant. The rules are defined in [`rules.md`](rules.md); this document describes the code that implements them.

* **Goal:** an agent that plays Joe better than strong human players. The current baseline is a rule-based `HeuristicAgent`; a `ProbabilisticAgent` chooses moves by expected value; the next step is a `BayesianAgent` that tracks beliefs about hidden cards (Section 7.4).
* **Long-term target:** a Godot game (iOS, Android and web) where humans and bots play together. The Python code is the reference implementation: the Godot version is expected to reimplement the engine, using `rules.md`, the action types (Section 5) and the test scenarios (Section 9) as its specification.
* **Requirements:** Python 3.13+, `numpy`, `python-statemachine`, `tqdm`. Tests use `pytest`.

---

## 2. Repository Layout

| File | Purpose |
| :--- | :--- |
| `rules.md` | The rules of Joe. The authority on game behaviour. |
| `cards.py` | `Suit`, `Rank`, `Card` and the 104-card double deck. |
| `config.py` | `JoeConfig`: penalty points, round objectives, round cap. |
| `player.py` | `Player`: a hand, score and round status. |
| `game_context.py` | `GameContext`: the full game state and every rule that changes it. |
| `actions.py` | The action types a player can take (Section 5). |
| `engine_driver.py` | `EngineDriver`: plays a game with action objects on either engine. |
| `engine.py` | The turn flow as a `python-statemachine` state machine (readable reference). |
| `fast_engine.py` | The same turn flow written by hand for speed. |
| `agents.py` | `RandomAgent`, `HeuristicAgent`, `ProbabilisticAgent`, planned `BayesianAgent`. |
| `evaluate_arena.py` | Plays full games between agents and reports results for agent 0. |
| `run_arena.py` | Command-line entry point for the arena. |
| `probabilistic_agent_theory.md` | The maths behind `ProbabilisticAgent`. |
| `tests/` | The test suite (Section 9). |

---

## 3. Game State (`GameContext`, `Player`, `JoeConfig`)
`GameContext` is the single source of truth for one game. Every card move goes through it, and it applies every rule; the engines only decide whose decision it is.

### 3.1 Cards
* A `Card` has a `suit` (0-3: Spades, Hearts, Clubs, Diamonds), a `rank` (0-12: Ace, Two, …, King) and a `deck_index` (0 or 1). Its `id` (0-103) is unique across the double deck.
* The two copies of a card are interchangeable under the rules, so actions and melds identify cards by suit and rank only.

### 3.2 Count Tensors
Many fields are `4 x 14` count arrays: rows are suits, columns are positions `0 = Ace (low), 1 = Two, …, 12 = King, 13 = Ace (high)`. An Ace is counted in **both** column 0 and column 13, so runs can use it at either end; `GameContext._sync_ace` adds or removes a card while keeping the two columns in step.

### 3.3 State Held by `GameContext`
| Field | Meaning |
| :--- | :--- |
| `deck` | The stock; the top card is the end of the list. |
| `discard_pile` | Discards; the top card is the end of the list. |
| `top_discard_available` | Whether the top discard can be taken. False once it has been taken and the card beneath exposed (that card stays buried, `rules.md` Section 4). |
| `top_discarder_idx` | Who discarded the top card; `None` for the opening upcard. |
| `players` | `Player` objects (Section 3.4). |
| `table_sets`, `table_runs` | Count tensors of the cards in sets and in runs on the table. |
| `runs_on_table` | Each run as a `TableRun(suit, low, high)` record, by end positions. Runs grow only at their ends. |
| `dead_cards` | Count tensor of discards that can no longer be taken (everything except an available top card). |
| `player_discard_counts`, `player_pickup_counts` | Per player, how many times each card was discarded / picked up this round (pickups include May-Is). Capped at 10. |
| `active_player_idx`, `dealer_idx`, `current_round_idx` | Whose turn it is, who dealt, and the round (0-6). |
| `total_actions` | Turns started this round (the name is historical). `current_circuit` is this divided by the number of players. |
| `may_i_target_idx`, `stock_draw_pending` | The May-I offer in progress, and whether the active player still has to draw from the stock. |
| `last_round_result`, `last_round_hands`, `last_round_down`, `last_round_circuits` | How the last round ended (`'out'`, `'stalemate'` or `'cap'`) and its final hands, kept because the next deal happens immediately. |

### 3.4 `Player`
Each hand is stored twice and kept in step: `hand_list` (the `Card` objects) and `private_hand` (a count tensor for fast meld searches). A player also has `score`, `is_down` and `may_is_used`.

### 3.5 `JoeConfig`
* Penalty points: `points_ace` (20), `points_eight_to_king` (10), `points_two_to_seven` (5).
* `objective_map`: round index → (sets needed, runs needed), e.g. `0: (2, 0)` for `[3, 3]`.
* `max_turns` (100): an engine safety net, not a rule. A round still running after this many circuits is ended and scored normally.

---

## 4. Rules Implemented in `GameContext`

### 4.1 Melds
A meld is a tuple:
* `('run', suit, start)`: 4 cards of `suit` at positions `start..start+3` (0 = low Ace, 13 = high Ace).
* `('set', rank)`: 3 cards of `rank`. Which suits doesn't matter, because any leftover card of that rank can go straight onto the set.

Melds are minimum size; longer melds are built by placing Key Cards afterwards.
* `find_all_melds(tensor, sets, runs)` lists every distinct combination meeting an objective; `find_melds` returns the first one. Both use the same search.
* `check_hand_objective(player_idx)`: whether a hand holds the round objective.
* `go_down(player_idx, melds)` lays the chosen melds, checking they exactly meet the objective with cards in hand (runs are worked out before sets, so a set never takes a card a run needs). It raises `ValueError` and changes nothing otherwise.

### 4.2 Placing Key Cards
* `placement_destinations(suit, rank)` lists where a card can go: `Destination.ON_SET`, `RUN_LOW_END`, `RUN_HIGH_END`. Choosing between two sets of the same rank, or between two runs sharing the same end, never changes what can be placed later, so these three cover every choice that matters.
* `is_key_card(suit, rank)`: whether a card fits anywhere.
* `execute_table_play` places a card at the chosen destination.
* `auto_place_cards` goes down (first valid melds) and places every Key Card, preferring runs. It is used by agents' look-ahead.

### 4.3 Turn Mechanics
* **Deal:** 11 cards each and the opening upcard. The first dealer of a game is random (`choose_first_dealer`); the deal then passes clockwise.
* **Pickup:** `execute_pickup_discard`, or `choose_stock`, which opens the May-I offer **before** the active player draws (`draw_pending_stock_card` gives them the card once the May-I is resolved).
* **May-I:** the offer goes clockwise from the player after the active player back round to them, skipping the discarder of the target card, players who are down, and players who have used 3 May-Is. A caller takes the discard plus the top stock card (`execute_may_i_call`).
* **Stock exhaustion:** the stock is rebuilt from the buried discards straight after the draw that empties it.
* **Discard:** `execute_discard` buries the previous top card unless it was already buried.

### 4.4 Round End and Scoring
* A round ends when a player goes out (no cards left), in a **stalemate** (every player down and the table closed: `is_stalemate`, `table_is_closed`), or at the round cap (`round_cap_reached`).
* `calculate_scores()` adds each player's hand penalty (`hand_penalty`) to their score, or nothing for anyone after a stalemate, records `last_round_result`, and advances `current_round_idx`.
* The game ends after 7 rounds; the lowest total score wins.

### 4.5 Other Interfaces
* `clone()`: a deep copy for agents' look-ahead.
* `get_oracle_truth(player_idx)`: the opponents' actual hands as a `(3, 4, 14)` tensor (next player clockwise first; zeros for the missing third opponent in 3-player games). It exists to measure how accurate an agent's beliefs are; agents must not use it to play.

---

## 5. Actions (`actions.py`)
Actions are small read-only records that compare by value and print readably (e.g. `Place 7♠ on the high end of a run`).

| Decision (`state_id`) | Who decides | Legal actions |
| :--- | :--- | :--- |
| `pickup_decision` | Active player | `PickStock()`; `PickDiscard()` when the top discard is available |
| `may_i_decision` | Player being offered the May-I | `CallMayI()`, `Pass()` |
| `go_down_decision` | Active player (only reached when the objective is held) | one `GoDown(melds)` per way to meet the objective, plus `Wait()` |
| `table_play_phase` | Active player (once down) | `Place(suit, rank, destination)` for every Key Card and destination, plus `EndTablePlay()` |
| `discard_phase` | Active player | `Discard(suit, rank)` for each card in hand |

* `GameContext.legal_actions(player_idx, state_id)` returns the list for a decision. Each card appears once, however many copies are held.
* `GoDown` stores its melds sorted, so the same melds in any order are the same action.

---

## 6. Engines

### 6.1 Turn Flow
Both engines implement the same states. Waiting states (decisions) are marked ★.

```
setup → dealing → start_turn → pickup_decision★
  ├─ PickDiscard ─────────────────────────────┐
  └─ PickStock → may_i_check ⇄ may_i_decision★ ┴→ processing_pickup (stock card drawn here)
processing_pickup
  ├─ already down        → table_play_phase★ ⇄ processing_table_play
  ├─ objective held      → go_down_decision★ → (GoDown) going_down → table_play_phase★
  └─ otherwise / Wait    → discard_phase★
table_play_phase★ → (EndTablePlay) discard_phase★, or round_end if the hand is empty
discard_phase★ → processing_discard → start_turn, or round_end (hand empty / round cap)
Going down or placing a card → round_end at once on a stalemate
round_end (scores) → dealing, or game_over after round 7
```

* **`engine.py`** uses `python-statemachine` and is the readable reference (`python engine.py` draws the state diagram; needs Graphviz).
* **`fast_engine.py`** implements the same flow by hand for speed and is used by the arena.
* Parity tests play the same games on both and require identical results.

### 6.2 Driving a Game (`EngineDriver`)
Both engines inherit `EngineDriver`:
* `advance()` steps through the states that need no decision (setup, dealing, turn start).
* `decider()` returns `(state_id, player_idx)` for the decision being waited on, or `None` when the game is over.
* `apply(action)` checks the action is in `legal_actions`, plays it, and advances to the next decision. An illegal action raises `ValueError` and changes nothing.

```python
ctx = GameContext(num_players=4)
engine = JoeEngine(ctx)
engine.advance()
while (decision := engine.decider()) is not None:
    state_id, player_idx = decision
    legal = ctx.legal_actions(player_idx, state_id)
    engine.apply(agents[player_idx].select_action(state_id, ctx, player_idx, legal))
```

---

## 7. Agents (`agents.py`)
Every agent implements `select_action(state_id, ctx, player_idx, legal)` and returns one of `legal`.

### 7.1 `RandomAgent`
Picks a legal action at random.

### 7.2 `HeuristicAgent` (baseline)
Rule-based play meant to resemble a basic human player:
* **Pickup:** take the discard if it fits the table (when down or holding the objective) or helps the objective.
* **May-I:** call if the discard completes another meld towards the objective; from Round 4 on, also if it is useful and the hand has 11 cards or fewer.
* **Go down:** when it leaves 5 or fewer unplaceable cards, otherwise with a probability that rises with time and with opponents' threat. The melds chosen are those leaving the lowest penalty in hand after placing Key Cards (ties at random).
* **Table play:** place every Key Card, preferring runs (chosen at random) to sets.
* **Discard:** the card with the lowest synergy score.

### 7.3 `ProbabilisticAgent`
Chooses pickups, May-Is, going down and discards by expected value in penalty points: the chance of completing the objective (from draw probabilities over the turns expected to remain) weighed against the chance an opponent ends the round first. Table play is inherited from `HeuristicAgent`. See [`probabilistic_agent_theory.md`](probabilistic_agent_theory.md).

### 7.4 `BayesianAgent` (planned)
Not implemented yet; the class docstring holds the design. In outline:
* **Belief tensor:** suits × ranks × (players + stock + discard pile + table). Each cell is the expected number of copies of a card in that place; each card's copies sum to 2 and each player's layer sums to their hand size.
* **Updates:** opponents' pickups and discards update their layers through per-objective likelihood matrices (52 × 52), followed by rebalancing. The matrices are to be tuned by optimisation.
* **Decisions:** score the 208 possible sets and 44 possible runs by how likely the agent is to complete them, and use that for pickups, discards, May-Is, going down and placing Key Cards (including deciding when to unlock or deny opponents).

---

## 8. Evaluation Arena (`evaluate_arena.py`, `run_arena.py`)
```
uv run python run_arena.py --mode {benchmark,probabilistic} --games 100 --players 4
```
Agent 0 (the challenger) plays against `HeuristicAgent`s over full 7-round games, in parallel across CPU cores. Results are from agent 0's point of view:

| Metric | Meaning |
| :--- | :--- |
| Tournament Win Rate | Games where agent 0 finished with the lowest score (ties count). |
| Round Win Rate | Rounds agent 0 won by going out. |
| Avg Point Diff | Agent 0's final score minus the opponents' average. |
| Avg Turns / Circuits per Round | Round length. |
| Down and Out Wins | Round wins where agent 0 went down and out on the same turn. |
| Strategic Detonations | Rounds another player won straight after agent 0's discard, without discarding themselves, where agent 0 was ahead of the winner when it discarded and lost fewer points than the other losers. |
| Stalemates / Rounds Hit Turn Cap | Rounds ended by a stalemate or by the round cap. |

`TournamentConfig(log_round_caps=True)` writes the final hands of capped rounds to `round_cap_tracker.log`.

---

## 9. Testing and Development Practice
* Run the suite with `uv run --group dev pytest`, and lint with `uv run --group dev ruff check .`.
* **Rules first:** `tests/test_rules_compliance.py` states each rule from `rules.md` and checks both engines obey it, using public behaviour (legal actions, `apply`, card movement) wherever possible. A rule deviation found before it can be fixed gets a test marked `pytest.mark.xfail(strict=True)`; the fix must remove the marker.
* **Test-driven changes:** write or update the test that describes the intended behaviour before changing the code.
* **Engine parity:** the two engines must play identical games (`tests/test_engine_parity.py`, `tests/test_engine_driver.py`).
* **Behaviour-preserving refactors:** when a change should not alter play, compare fixed-seed full games before and after; they must be identical.
* **Other test files:** `test_actions.py` (action types, legal actions), `test_game_context.py` (state and melds), `test_joe_engine.py` (individual transitions), `test_agents.py` (agent decisions), `test_evaluate_arena.py` (arena metrics), plus `test_cards.py`, `test_player.py`, `test_config.py`.
* **Portability:** keep the engine deterministic for a given random seed, and express new decisions as action types, so the Godot implementation can follow the same specification and reuse the test scenarios.
