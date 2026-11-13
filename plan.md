# Development Plan: JoeNet Actor-Critic

## Overview
This document strictly dictates the step-by-step Test-Driven Development (TDD) sequence for the JoeNet architecture. 
**Rule of Execution:** Do not jump ahead. A step must be fully implemented, with all tests passing, before moving to the next. All legacy CFR logic must be aggressively stripped during the refactoring steps.

---

## Phase 1: Environment & Engine Refactoring (The Purge)
*Goal: Audit and port the legacy game environment, ensuring it perfectly aligns with the new data layer requirements and is free of CFR-specific trackers.*

* **Step 1.1: Core Configuration & Entities (`config.py`, `card.py`, `player.py`)**
    * [x] Update `JoeConfig` to include the new asymmetric terminal scoring multipliers (`catch_up_multiplier`, `pull_ahead_multiplier`).
    * [x] Refactor `Player` to ensure it cleanly separates Logic State (Card objects) from NN State (pre-computed Numpy arrays).
* **Step 1.2: The Ground Truth (`game_context.py`)**
    * [x] *Action:* Strip out any old CFR regret tracking or history tree logic.
    * [x] *Feature:* Implement and test the `get_oracle_truth(active_player_idx)` method to return the exact `(3, 4, 14)` tensor of opponent hands with strict zero-padding for 3-player games.
* **Step 1.3: The Dual Engines (`engine.py` & `fast_engine.py`)**
    * [x] *Action:* Strip out all Monte Carlo rollout hooks or CFR node tracking.
    * [x] *Feature:* Verify state transitions and ensure the game strictly ends the moment a hand is empty. 

---

## Phase 2: Tensor Generation & The Information Set
*Goal: Build the translation layer that converts the Python game state into strictly formatted NumPy tensors for the neural networks.*

* **Step 2.1: Spatial Tensor Builder**
    * [x] Write tests for exactly 13 channels representing the public board (Suit/Rank dimensions: 4x14, handling duplicate wrap-around Aces).
    * [x] Ensure 3-player zero-padding is mathematically perfect for missing opponent channels.
* **Step 2.2: Scalar Tensor Builder**
    * [x] Write tests for the 28-feature 1D array (phases, stock depth, scores).
    * [x] Implement the `Is 3-Player Game?` flag and verify zero-padding for the missing opponent's scalars.
* **Step 2.3: Action Space & Masking Generator**
    * [x] Implement the dynamic 58-logit masking logic based on the `Active Phase`. Ensure illegal moves are strictly zeroed out.

---

## Phase 3: The Reward Layer (PBRS Math)
*Goal: Implement the deterministic Potential-Based Reward Shaping math to resolve the Credit Assignment Problem.*

* **Step 3.1: Combinatorial Outs Calculator (`Ukeire`)**
    * [x] Write tests to calculate required cards minus known dead cards, discounting dynamically based on a mocked `oracle_probs` tensor.
* **Step 3.2: Danger Score Calculator (`Betaori`)**
    * [x] Write tests for calculating the `Relative Deadwood Margin`.
    * [x] Verify the math allows the Danger Score to safely invert into a positive Potential bonus (the detonation strategy).
* **Step 3.3: Asymmetric Terminal Scoring**
    * [x] Implement the terminal score calculator using the `catch_up` (2.0x) and `pull_ahead` (0.5x) multipliers. 

---

## Phase 4: Neural Network Architecture (PyTorch)
*Goal: Build the deep learning models ensuring strict dimensional contracts and the Oracle concatenation trick.*

* **Step 4.1: Shared Representation Heads**
    * [x] Build the `SpatialHead` (CNN with asymmetric branches for Runs vs Sets) and `ScalarHead` (MLP).
* **Step 4.2: OracleNet & The Concat Trick**
    * [x] Build `OracleNet` to output the `(B, 3, 4, 14)` sigmoid probabilities.
    * [x] Implement the exact channel concatenation to create the 16-channel Expanded Spatial Tensor.
* **Step 4.3: ActorNet & CriticNet**
    * [x] Build `CriticNet` (outputting `(B, 1)` linear EV).
    * [x] Build `ActorNet` (outputting `(B, 58)` masked logits).

---

## Phase 5: Phase 1 & 2 Execution (Static Pre-Training)
*Goal: Generate the foundational ground truth and run Behavioral Cloning.*

* **Step 5.1: The Heuristic Agent**
    * [x] Build the hardcoded bot, including a "panic" threshold to prevent permanent sandbagging from polluting the dataset.
* **Step 5.2: Data Generation (HDF5 Buffer)**
    * [x] Build the high-speed writer to record `spatial`, `scalar`, `action_mask`, `oracle_truth`, `terminal_score`, and `policy` arrays. 
    * [x] Run Phase 1 sandbox data generation.
* **Step 5.3: Supervised Training Loop (Phase 2)**
    * [x] Build the PyTorch Dataloaders and the backpropagation loop for Behavioral Cloning (Actor) and Monte Carlo regression (Critic).

---

## Phase 6: The Arena (Evaluation & Benchmarking)
*Goal: Build the tournament runner to accurately measure macro-level performance. This MUST be built before RL begins so we can benchmark the Phase 2 cloned agent against the baseline.*

* **Step 6.1: Agent Wrappers**
    * [x] Wrap the `RandomAgent` (acts purely on masked logits).
    * [x] Wrap the `HeuristicAgent` (acts on hardcoded rules).
    * [x] Wrap the `JoeNetAgent` (acts on trained neural weights).
* **Step 6.2: The Tournament Runner**
    * [x] Build a lightweight simulation wrapper that sequentially executes a full 7-round game, tracking running scores across the phase transitions.
* **Step 6.3: Metric Tracking & Logging**
    * [x] Track `Tournament Win %`, `Round Win %`, and `Average Point Differential`.
    * [x] Implement a tracker for "Strategic Detonations" (intentional early round ends).
* **Step 6.4: The Baseline Evaluation**
    * [x] Run the Phase 2 Cloned agent through the Arena to verify it successfully learned the baseline heuristics.

---

## Phase 7: Reward Layer Refactor (Dynamic PBRS)
*Goal: Discard abstract heuristic scoring and build a stateless EV math engine in `reward.py`, fed dynamically by tensor deserialization in `buffers.py`.*

* **Step 7.1: The Dense Reward Engine (`reward.py`)**
    * [ ] Create `StateEvaluator` class. Migrate the pure math (`_evaluate_hand_state`, `_calculate_avalanche_threat`, etc.) completely out of `agents.py`.
    * [ ] Delete obsolete `RewardCalculator` abstract heuristic logic (Ukeire/Betaori).
    * [ ] Retain `calculate_asymmetric_score` to anchor the terminal state.
* **Step 7.2: Tensor Deserialization (`buffers.py`)**
    * [ ] Build a robust helper function to map the 13-channel spatial tensor and scalar array back into a mocked `GameContext`.
    * [ ] Ensure correct slicing (e.g., mapping Channel 1 to `ctx.table_sets`, Channels 8-11 to `ctx.player_pickup_counts`).
* **Step 7.3: Dynamic PBRS Integration**
    * [ ] Update the PyTorch `Dataset` to call the deserializer from `buffers.py`, passing the mocked context into the `reward.py` evaluator.
    * [ ] Write unit tests for the $F(s,a,s') = \Phi(s') - \Phi(s)$ calculation, ensuring $\Delta EV$ accurately rewards positive hand progression and penalizes accumulating deadwood.

---

## Phase 8: Phase 3 & 4 Execution (RL Self-Play)
*Goal: Implement TD Learning using the newly wired Probabilistic dense rewards.*

* **Step 8.1: TD Target & Rollout Buffer Update**
    * [ ] Update the temporary Rollout Buffer to calculate the TD Error using the injected $\Phi(s)$ and $\Phi(s')$ values from the math engine.
* **Step 8.2: Phase 3 (Exploratory Training Loop)**
    * [ ] Implement entropy injection for forced exploration.
    * [ ] Execute the Actor-Critic reinforcement training loop using the new dense step rewards.
* **Step 8.3: Phase 4 (Mastery Training Loop)**
    * [ ] Remove exploration constraints and finalize evaluation parameters.
    * [ ] Run the final JoeNet model through the Arena to prove superhuman tournament performance against the baseline Heuristic and Probabilistic agents.