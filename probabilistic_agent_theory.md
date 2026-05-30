# ProbabilisticAgent: Mathematical Architecture

The `ProbabilisticAgent` operates as a pure frequentist mathematician. It evaluates every possible action by simulating future realities and grading them based on a rigorous Expected Value (EV) calculation. It relies entirely on probability formulas and avoids arbitrary heuristics.

---

## 1. The Master Formula (`_evaluate_hand_state`)

The Master Equation calculates the literal Expected Value (EV) of any given hand state in terms of penalty points. 

It evaluates two distinct, independent timelines:
* **Timeline A (The Avalanche Threat):** The game ends *this exact turn* (1-Turn Horizon). The bot takes its current physical penalty.
* **Timeline B (The Future Trajectory):** The game continues. The bot relies on its cumulative probability of winning ($P_{win}$) over the remaining horizon to reduce its deadwood.

### The Pairwise Difference EV Calculation
To prevent the bot from becoming mathematically "greedy" in 4-player games, the EV uses a **Pairwise Difference multiplier**. It multiplies the bot's personal risk by the number of opponents to restore a strict $1:1$ risk/reward ratio.

**Variables:**
* $N$: Number of opponents (`num_opponents`).
* $P_{lose}$: The 1-turn sudden death probability (`p_lose`).
* $P_{win}$: The cumulative probability of completing the round objective (`p_win`).
* $Pen_{current}$: Total points in hand (or just deadwood if already down).
* $Pen_{future}$: $(P_{win} \times \text{Deadwood}) + ((1.0 - P_{win}) \times \text{Total Hand Points})$.
* $OppPen$: The estimated total penalty points currently held by all opponents.

**The Math:**
$$Risk = N \times ((P_{lose} \times Pen_{current}) + ((1.0 - P_{lose}) \times Pen_{future}))$$

$$Reward = (1.0 - P_{lose}) \times (P_{win} \times OppPen)$$

$$EV = Reward - Risk$$

Because the goal is survival, the bot always seeks the action that yields the **highest EV** (which mathematically translates to the least negative penalty risk).

---

## 2. Subsidiary Math Engines

### The Hypergeometric Engine (`_get_draw_probability`)
Calculates the cumulative probability of drawing the required pieces of a meld over the remaining turns ($H$). 

**The Math (Binomial Approximation):**
It uses a Binomial approximation for speed, calculating $P(X \ge k) = 1.0 - P(X < k)$.
$$p = \frac{\text{Available Copies}}{\text{Unknown Deck Size}}$$
$$P(X < req) = \sum_{k=0}^{req-1} \left( \binom{H}{k} \cdot p^k \cdot (1-p)^{H-k} \right)$$
$$P_{draw} = 1.0 - P(X < req)$$

### The Dynamic Greedy Algorithm (`_find_best_seed_allocation`)
Determines the absolute best $P_{win}$ for a given hand. A "seed" is defined as any single held card (or contiguous block) that contributes to an objective.

**The Math:**
1. Parse all valid seeds from the hand tensor.
2. Calculate the exact $P_{draw}$ for the missing distance of each seed.
3. Greedily multiply the highest probabilities together: $P_{win} = \prod P_{best\_seeds}$
4. **Probabilistic Smoothing:** If requirements are entirely missing (e.g., 0 cards toward a run), it applies the "Miracle Draw" odds for pulling it from scratch over the remaining horizon:
$$P_{win} = P_{win} \times (P_{draw\_miracle})^{\text{missing\_reqs}}$$

### The Ground-Truth Outs (`_get_available_tensor`)
Maintains a rigorous tracking of the unknown deck to cure the "Average Card Fallacy."

**The Math:**
The environment strictly plays with a double deck (max 2 copies of any card).
$$Available = 2 - (Table + Discards + MyHand + KnownOpponentHands)$$

### The Sudden Death Radar (`_calculate_avalanche_threat`)
Calculates $P_{lose}$, the threat that an opponent ends the round before the bot's next turn. It evaluates two distinct states for each opponent:

**The Math (If Opponent is DOWN):**
They can use table "sparks" to trigger an avalanche.
$$P_{spark} = \frac{\text{Physical Table Outs}}{\text{Unknown Deck Size}}$$
$$P_{hold} = \left( \frac{\text{Opponent Hand Size}}{\text{Unknown Deck Size}} \right)^{\text{Opp Hand Size} - 1}$$
$$Threat_{down} = P_{spark} \times P_{hold}$$

**The Math (If Opponent is NOT DOWN):**
They cannot use table sparks. Threat is purely a function of the ticking clock nearing the empirical end of the round.
$$Threat_{active} = 0.05 \times \left( \frac{\text{Current Circuit}}{\text{Expected Horizon}} \right)$$

$$P_{lose} = 1.0 - \prod (1.0 - Threat_{opp})$$

---

## 3. Action Decisions (The 1-Ply Stochastic Lookahead)

The bot's `select_action` method completely overrides static heuristic rules. It uses the Master Equation to mentally simulate the future exactly one step ahead for every available move.

### The Discard Decision
The bot creates a hypothetical tensor for every legally discardable card in its hand, removes it, and calculates the resulting EV.
* **Logic:** Select card where $EV_{hypothetical}$ is maximized.

### The Pickup Decision (Face-Up vs. Stock)
The bot simulates picking up the face-up discard, and then simulates its *best possible discard* from that new hand. It compares this against rolling the dice on a blind stock draw.
* **Math:** $EV_{stock} = EV_{baseline} - (\text{Average Stock Point Value} \times P_{lose})$
* **Logic:** `IF (best_post_discard_ev > EV_stock)` $\rightarrow$ Pickup Face-Up Card.

### The May-I Decision (Time-Dilated EV)
The bot treats a May-I as an **"Extra Turn."** It evaluates picking up the discard while artificially expanding its search space, naturally spiking the $P_{win}$ reward. 
* **Math:** Calculate $EV_{pickup}$ using $Horizon\_Offset = +1$.
* **Math:** Calculate the pure sudden-death cost of the blind penalty card: $Cost = \text{Average Stock Point Value} \times P_{lose}$
* **Logic:** `IF (EV_{pickup} - Cost > EV_{baseline})` $\rightarrow$ Take the May-I.

### The Go Down Decision (Tactical Sandbagging)
The bot simulates two futures: Waiting vs. Physically putting melds on the table. If it puts cards on the table, its deadwood drops to zero, but it creates "sparks" that allow opponents to legally play their own cards, causing the opponent `Avalanche Threat` ($P_{lose}$) to spike. 
* **Logic:** `IF (EV_{waiting} > EV_{going\_down})` $\rightarrow$ Keep melds hidden in hand.

# BayesianAgent: Mathematical Architecture

The `BayesianAgent` is an advanced player-profiling engine that evolves the foundational Expected Value (EV) math of the `ProbabilisticAgent`. 

While the baseline probabilistic engine relies on uniform hypergeometric distribution (assuming all unknown cards are equally likely to be in the deck), the `BayesianAgent` uses **Bayesian Inference** and an **Observation Tensor** to build a dynamic probability heatmap of the opponents' hidden hands.

---

## 1. The Core Paradigm Shift: Curing the Average Card Fallacy

In standard frequentist math, if there are 50 unknown cards and 2 copies of the $8\heartsuit$, the probability of drawing the $8\heartsuit$ from the deck is $2/50$ ($4\%$). 

The `BayesianAgent` recognizes that a game of Joe is not truly random. Opponents are heuristic actors. If an opponent picks up the $7\heartsuit$ from the discard pile, the deck is no longer uniform. The probability that the opponent is hoarding the $8\heartsuit$ skyrockets, meaning the probability that the $8\heartsuit$ is sitting in the Stock Pile plummets.

The `BayesianAgent` cures the "Average Card Fallacy" by evaluating the deck as a **filtered subset**, not a random subset.

---

## 2. The Observation Tensor (The "Scent" Map)

Instead of just tracking what cards have been explicitly revealed, the bot maintains a persistent, per-opponent Observation Tensor (`shape: 4, 14`). This tensor stores the probability ($0.0$ to $1.0$) that an opponent is actively holding or seeking a specific card.

### The Propagation Math (Spreading the Scent)
When an opponent picks up a face-up card (e.g., $7\heartsuit$), the bot does not just register the $7\heartsuit$. It uses a **Radial Basis Function (RBF)** to propagate "scent" to mathematically adjacent cards.

* **Run Synergy (Same Suit, Adjacent Ranks):**
  $$P_{scent}(Rank \pm 1) = \alpha \cdot \text{Base Scent}$$
  $$P_{scent}(Rank \pm 2) = \beta \cdot \text{Base Scent}$$
* **Set Synergy (Same Rank, Different Suits):**
  $$P_{scent}(Other Suits, Same Rank) = \gamma \cdot \text{Base Scent}$$

*(Where $\alpha, \beta, \gamma$ are decay weights based on distance and the round's objective).*

### Bayesian Update Rule (Double-Deck Reality)
When new evidence ($E$) is observed (a discard or a pickup), the bot updates its belief that an opponent is building a specific meld Hypothesis ($H$).
$$P(H | E) = \frac{P(E | H) \cdot P(H)}{P(E)}$$

Because the game uses a double deck, a discard is "noisy evidence." If an opponent discards the $9\spadesuit$, it is highly likely they are not building Spades. However, there is a small chance they *are* building Spades and simply discarding a toxic duplicate. 

Therefore, $P(E | H_{SpadesRun})$ does not drop to $0.0$; it drops to the probability of drawing a duplicate ($\approx 0.05$). The bot mathematically heavily discounts the Spades run hypothesis, but retains a residual "Ghost Scent" to prevent being completely blindsided by a duplicate-venting opponent. 

Crucially, the scent for the *specific discarded card* ($9\spadesuit$) is zeroed out entirely, as the opponent mathematically cannot need it anymore.

---

## 3. The Heatmap Engine (Overriding `_get_available_tensor`)

The `BayesianAgent` intercepts the standard `_get_available_tensor` function. Instead of returning physical integer counts (e.g., "There are 2 copies left"), it returns a **Weighted Probability Matrix**.

**The Math:**
$$P_{deck\_location} = 1.0 - \sum_{i=1}^{N_{opps}} P_{scent}(Opponent_i, Card)$$

$$Available_{Bayesian} = Available_{Physical} \times P_{deck\_location}$$

**Example:**
There are physically 2 copies of the $J\clubsuit$ unplayed. 
However, Opponent 1's scent map shows a $0.85$ probability they are holding them.
$$Available_{Bayesian} = 2 \times (1.0 - 0.85) = 0.30$$

When the Dynamic Greedy Algorithm (`_find_best_seed_allocation`) asks for the probability of completing a $J$ Set, the math engine evaluates it as if there are only **$0.30$ copies** left in the deck, not $2$.

---

## 4. Strategic Alterations to the Master Equation

By feeding this Bayesian Heatmap down into the established EV Math, the bot gains two massive behavioral upgrades against the 4-Player Swarm without altering a single line of the `select_action` logic.

### 1. Identifying "Toxic Seeds" (Defensive Pivot)
In the 1-Ply Lookahead, if the bot holds a $4\diamondsuit$ and $5\diamondsuit$, the baseline bot would aggressively chase the $6\diamondsuit$. 
The `BayesianAgent` checks the heatmap, sees an opponent is actively radiating scent around the $6\diamondsuit$, and calculates the true $P_{draw}$ as $< 1\%$. The $4\diamondsuit$ and $5\diamondsuit$ instantly become mathematically worthless ("Toxic Seeds"). The EV calculation drops, and the bot immediately discards them to avoid deadwood, seamlessly pivoting to a safer meld.

### 2. Starving the Swarm (Predatory Discarding)
When calculating the EV of its discard options, the bot evaluates the resulting $P_{lose}$ (Avalanche Threat). 
Because the Rummy deck is now mapped as a probability field, the bot knows exactly which cards the opponents need. 
$$Threat_{discard} = \sum_{i=1}^{N_{opps}} P_{scent}(Opp_i, Discard\_Card)$$
If the bot considers discarding a card with a massive scent weight, the simulated $P_{lose}$ spikes artificially high. The Master Equation's Pairwise Multiplier reacts violently to this risk, mathematically forcing the bot to **hoard the spark in its own hand**, intentionally starving the opponents of their required connections and dragging the game out into a longer horizon where the bot's EV logic can reliably dominate.