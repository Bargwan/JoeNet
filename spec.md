# Specification: JoeNet (Actor-Critic & Belief State)

## 1. Project Overview
* **Goals:** Develop a high-performance, superhuman AI agent for the imperfect-information card game **Joe** using a modular **Deep Reinforcement Learning (Actor-Critic)** architecture, augmented by an **Oracle (Belief State)** network.
* **Core Mechanisms:** The AI utilizes **Temporal Difference (TD) Learning** to evaluate states dynamically. Sparse terminal rewards are decomposed into dense, step-by-step signals using mathematically rigorous **Potential-Based Reward Shaping (PBRS)**.
* **Performance Target:** The AI must execute inference in a fraction of a millisecond, making it suitable for direct, real-time integration into a Godot game engine without requiring simulation pauses.

---

## 2. Software Architecture
The engine consists of four primary layers, cleanly decoupling the rules of the game from the neural logic that plays it.

**⚠️ Critical Note on Legacy Code Porting:** Some foundational files ported into this project (e.g., `game_context.py`, `fast_engine.py`) were originally designed for a Counterfactual Regret Minimization (CFR) architecture. If any legacy CFR-specific logic (e.g., tree traversal trackers, regret matching utilities, or Monte Carlo rollout hooks) is encountered, it must be aggressively stripped out. Do not attempt to integrate CFR mechanisms into this Actor-Critic architecture.

### 2.1 The Data Layer (`GameContext` & `Player`)
* **GameContext:** A purely data-driven class containing the "Truth" of the game. It holds the deck, the discard pile, the list of `Player` objects, and the table melds. It contains **no rule enforcement logic**, only data manipulation methods.
    * **Ground Truth Interface:** Implements `get_oracle_truth(active_player_idx)`, which extracts the exact `private_hand` arrays of the three opponents and stacks them into a `(3, 4, 14)` tensor, representing the omniscient reality of the hidden cards.
* **Player:**
    * **Logic State:** A list of `Card` objects (maintaining unique IDs 0-103) for exact game logic.
    * **NN State:** Pre-computed `4x14` Numpy arrays (e.g., `private_hand`) for rapid tensor generation.
* **JoeConfig:** An injected configuration object that dictates scoring, round objectives, and RL hyperparameters (e.g., asymmetric terminal scoring multipliers).

### 2.2 The Logic Layer (Dual-Engine Design)
To balance code maintainability with the extreme execution speed required for Reinforcement Learning, the logic layer utilizes a dual-engine architecture.
* **`engine.py` (Development & Debugging):** Built using the `python-statemachine` library. It provides highly readable state transitions, clear guard logic, and the ability to auto-generate visual flowcharts of the game rules. It is the definitive reference for game logic.
* **`fast_engine.py` (Training & Inference):** A mathematically identical, hardcoded replacement. It is stripped of all Python reflection overhead and utilizes lightweight `FastState` objects, enabling instantaneous RAM cloning and massive throughput during RL data generation.
* **Responsibility:** Both engines enforce the strict chronological flow of the game. They query `GameContext` to validate guards and trigger state transitions.
* **End-of-Round Condition:** Authentic Joe rules are enforced. The round immediately ends and scores are tallied the moment any single player empties their hand.

### 2.3 The Neural Layer (Modular Networks)
To prevent gradient interference between vastly different mathematical tasks, the neural architecture is split into three distinct modules that share common Convolutional and MLP input heads via Transfer Learning.
* **Shared Heads:** `SpatialHead` (asymmetric CNN branches for Runs vs Sets) and `ScalarHead` (MLP for game rules).
* **OracleNet:** Predicts the exact hidden cards held by all opponents based on public history.
* **CriticNet:** Calculates the Expected Value (EV) of the current board state.
* **ActorNet:** Generates the 58-logit policy distribution for action selection.

### 2.4 The Reward Layer (`reward.py`)
Contains the deterministic mathematical logic required for Potential-Based Reward Shaping, evaluating the exact "Outs" and "Danger" of a hand to provide dense TD-learning signals without warping the game's ultimate Nash Equilibrium.

---

## 3. Input Tensors (The Information Set)
The Neural Networks process rigorously formatted tensors. Variables: **B** = Batch Size.

### 3.1 Spatial Tensor (Input for CNN Head)
**Base Shape:** `(B, Channels=13, Height=4, Width=14)`
**Data Type:** `numpy.int8` at generation, cast to `float32` during DataLoader collation.
* **Dimension 1 (4):** Suits (Spades, Hearts, Clubs, Diamonds).
* **Dimension 2 (14):** Ranks (A-Low, 2, 3, ..., K, A-High). *Aces are duplicated at index 0 and 13 to allow wrap-around run detection.*

| Channel Idx | Feature Name | Description | Values (`int8`) |
|:---:|:---|:---|:---|
| 0 | **Private Hand** | Cards currently held by the observing player. | `{0, 1, 2}` |
| 1 | **Table Sets (3s)** | Cards currently on the table in "Set" melds. | `{0, 1, 2}` |
| 2 | **Table Runs (4s)** | Cards currently on the table in "Run" melds. | `{0, 1, 2}` |
| 3 | **Discard Pile (Top)** | The literal top card of the discard pile (visible). | `{0, 1}` |
| 4-7 | **Discard History** | Presence counts of cards discarded by Self, Op1, Op2, Op3. | `{0, 1, 2}` |
| 8-11 | **Pickup History** | Presence counts of cards picked up by Self, Op1, Op2, Op3. | `{0, 1, 2}` |
| 12 | **Known Dead Cards** | Cards seen in discard pile that are now buried. | `{0, 1, 2}` |

### 3.2 Scalar Tensor (Input for MLP Head)
**Shape:** `(B, Features=28)`
**Data Type:** `numpy.float32`. All values are normalized to roughly `0.0` to `1.0`.
* Contains one-hot phase encodings, turn numbers, stock depth, player scores, hand sizes, and May-I counts.

### 3.3 Variable Player Counts (Zero-Padding)
The Neural Network architecture uses fixed-size tensors designed for the maximum case of 4 players (Self + 3 Opponents). 
To support 3-player games, the engine and buffers must strictly apply zero-padding to the missing "Op3" (Upstream/Right) player slot to maintain the exact tensor dimensions:
* **Spatial Tensor:** Channels for Op3 Discards and Pickups are set to a `4x14` matrix of zeros.
* **Scalar Tensor:** Features for Op3's Score, Hand Size, and May-I counts are set to `0.0`. The `Is 3-Player Game?` feature is set to `1.0`.
* **Oracle Truth & Probs:** The 3rd opponent channel in the `(B, 3, 4, 14)` tensor is strictly set to a matrix of zeros. The Actor/Critic network learns to mathematically ignore this channel when the `Is 3-Player Game?` scalar flag is active.

---

## 4. Neural Network I/O Contracts

### 4.1 OracleNet (Belief State Perception)
* **Task:** Predict the contents of the opponents' hidden hands based purely on public history.
* **Inputs:** `spatial_x` `(B, 13, 4, 14)`, `scalar_x` `(B, 28)`.
* **Outputs:** `oracle_probs` `(B, 3, 4, 14)`. Uses a Sigmoid activation to represent the independent 0.0 to 1.0 probability of each opponent holding specific cards.

### 4.2 The Concatenation Trick (Actor & Critic Vision)
Because `oracle_probs` shares the exact `4x14` spatial dimension as the public board, it is directly concatenated onto the base spatial tensor along the channel dimension. 
* **Expanded Spatial Tensor:** 13 Public Channels + 3 Oracle Channels = 16 Total Channels.
This allows the Actor and Critic's shared `SpatialHead` CNN to scan public cards and predicted hidden cards simultaneously using the exact same filters.

### 4.3 CriticNet (Value Estimation)
* **Task:** Estimate the Expected Value (EV) in terminal points from any given state.
* **Inputs:** `expanded_spatial_x` `(B, 16, 4, 14)`, `scalar_x` `(B, 28)`.
* **Outputs:** `expected_value` `(B, 1)`. A linear scalar representing the predicted final score delta.

### 4.4 ActorNet (Policy Formulation)
* **Task:** Select the optimal action.
* **Inputs:** `expanded_spatial_x` `(B, 16, 4, 14)`, `scalar_x` `(B, 28)`, `action_mask` `(B, 58)`.
* **Outputs:** `action_logits` `(B, 58)`. Masked logits defining the action distribution.

### 4.5 Action Space (State-Conditioned Overloading)
To keep the output head lean (58 Logits), the meaning of specific indices dynamically shifts based on the `Active Phase` one-hot feature in the Scalar Tensor.
* **Logits 0-1:** `[Pick_Stock, Pick_Discard]`
* **Logits 2-3:** `[Call_MayI, Pass]`
* **Logits 4-5:** `[Go_Down, Wait]` OR `[<Unused>, End_Table_Play]`
* **Logits 6-57:** "Move Card X" (Used for both Discarding and Table Play).

---

## 5. Potential-Based Reward Shaping (PBRS) & Terminal Scoring
To resolve the Credit Assignment Problem, the Critic network is trained via Temporal Difference (TD) learning using dense rewards. Sparse terminal rewards are decomposed into dense rewards using the formula:
$$F(s,a,s')=\Phi(s')-\Phi(s)$$

The Potential Function $\Phi(s)$ is strictly defined as:
$$\Phi(s)=(\text{Combinatorial Outs})-(\text{Danger Score})$$

### 5.1 Combinatorial Outs (Ukeire)
$$Effective\ Outs=Total\ Remaining-\sum_{i \in Opponents}P(Player_i\ holds\ card)$$
The engine calculates the required cards to fulfill objectives, subtracting dead cards and probabilistically discounting cards the `OracleNet` predicts opponents are holding.

### 5.2 Danger Score (Betaori)
$$Danger(card)=P(\text{Opponent Needs Card})\times(\text{Projected Asymmetric Margin})$$
The Projected Asymmetric Margin calculates the overall tournament score impact if the opponent goes out. It compares the agent's projected tournament score (Current Score + Hand Deadwood) against the opponent's projected tournament score (Current Score + 0), and scales the difference using the Asymmetric Multipliers. This allows the Danger Score to flip negative (becoming a positive Potential bonus) *only* if detonating on a trailing player mathematically secures the agent's tournament position.

### 5.3 Asymmetric Terminal Scoring
The absolute "Ground Truth" fed to the Critic at the end of a round is scaled to encode tournament awareness (Risk Aversion vs. Desperation). 
The raw point differential between the agent and opponents is multiplied by configurable parameters defined in `JoeConfig`:
* **Trailing (Catch-up margin):** Multiplied by `catch_up_multiplier` (default: **2.0**) to aggressively reward catching up.
* **Leading (Pull-ahead margin):** Multiplied by `pull_ahead_multiplier` (default: **0.5**) to devalue greedy plays when already winning.

---

## 6. Training Regimes (The Ground Truths)
The neural architecture transitions through distinct training phases, altering the "ground truth" targets for the Actor and Critic networks.

### 6.1 Phase 1: Data Generation (The Sandbox)
* **Goal:** Create a massive, foundational dataset representing basic, competent gameplay.
* **Mechanism:** The neural networks are completely inactive. A hardcoded `HeuristicAgent` plays millions of fast-paced games against itself. Every public board state, chosen action, hidden opponent hand, and final Asymmetric Terminal Score is recorded directly to an HDF5 buffer on disk.

### 6.2 Phase 2: Supervised Pre-Training (Behavioral Cloning)
* **Goal:** Bootstrap the neural networks with a baseline understanding of the rules, basic heuristics, and opponent hand prediction.
* **Mechanism:** The networks train purely on the static HDF5 dataset generated in Phase 1 using standard supervised learning (backpropagation).
* **Targets:**
    * **ActorNet:** Optimized via Cross-Entropy Loss to mimic the exact action probability distribution (`(N, 58)` policy array) of the Heuristic bot.
    * **CriticNet:** Optimized to predict the resulting Asymmetric Terminal Scores of those Heuristic games (Monte Carlo evaluation).
    * **OracleNet:** Optimized via Binary Cross-Entropy to predict the absolute ground-truth matrix of the hidden opponent hands recorded during those games.

### 6.3 Phase 3: Exploratory Reinforcement Learning (TD Learning)
* **Goal:** Break past the Heuristic bot's skill ceiling and un-learn its hardcoded flaws using Temporal Difference (TD) learning.
* **Mechanism:** The agent stops reading the static file and begins playing live games. The system forces a high degree of exploration so the agent can discover alternative, mathematically superior actions.
* **Targets:**
    * **Critic Target:** The Critic bootstraps its own predictions step-by-step using the hardcoded PBRS math (Combinatorial Outs - Danger Score), anchored by the terminal score at the exact end of the round.
    * **Actor Target:** The Actor abandons Behavioral Cloning and uses Policy Gradient methodologies to dynamically adjust its action distribution, aiming to strictly maximize the Expected Value predicted by the Critic.

### 6.4 Phase 4: Pure Self-Play Mastery
* **Goal:** Fine-tune superhuman strategies and completely exploit the learned mathematical realities of the game.
* **Mechanism:** Exploration constraints are minimized. The AI now fully trusts its Critic network and plays ruthlessly optimal moves against itself to solidify advanced tactics (such as the asymmetric detonation strategy).

---

## 7 HDF5 Buffer Schema
Training data is streamed directly to disk via `h5py`. Variables: **N** = Buffer Max Size.

| Dataset Name | Shape | Datatype | Description |
| :--- | :--- | :--- | :--- |
| `spatial` | `(N, 13, 4, 14)` | `int8` | The core public game state. |
| `scalar` | `(N, 28)` | `float32` | Game phase, turn counts, and stock depth. |
| `action_mask` | `(N, 58)` | `bool_` | The legal actions available at that specific state. |
| `oracle_truth` | `(N, 3, 4, 14)` | `int8` | Absolute ground-truth of the opponents' hidden hands. |
| `terminal_score` | `(N, 1)` | `float32` | Actual final point delta of the episode. |
| `policy` | `(N, 58)` | `float32` | Executed action probabilities (Behavioral Cloning). |

---

## 8. Development Methodology
* **Strict Test-Driven Development (TDD):** All features, tensor mappings, and mathematical functions must be developed test-first. No application code is to be written until a failing test confirms its necessity and boundary conditions.
* **Sequential Execution:** The development lifecycle will strictly follow a step-by-step sequence defined in an associated `plan.md` file. Components will be built and verified in complete isolation before moving to the next step of the pipeline. Do not jump ahead of the plan.