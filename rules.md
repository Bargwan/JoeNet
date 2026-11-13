# Rules of Joe

## 1. Terminology
* **Game Round:** One of the 7 deals that make up a game. Each Game Round has its own objective (Section 3). The word "round" on its own always means a Game Round.
* **Turn:** One player's Pickup → Action → Discard sequence (Section 4).
* **Meld:** A "3" (Set) or a "4" (Run) (Section 3).
* **Key Card:** A card in a player's hand that could legally extend a meld already on the table. Once it is placed on a meld it becomes part of that meld and is no longer a Key Card.
* **Going Down:** Placing the Game Round's objective melds from your hand face-up on the table (Section 6).
* **Going Out:** Ending your turn with no cards in your hand (Section 6).

## 2. Setup & The Deal
* **Players:** 3 or 4.
* **The Deck:** Two standard 52-card French decks shuffled together (104 cards). There are **no jokers or wild cards**.
* **The Deal:** Each player is dealt 11 cards face-down.
* **The Board:** The remaining cards are placed face-down to form the **Stock Pile**. The top card is turned face-up to start the **Discard Pile** (the **opening upcard**).
* **Turn Order:** Play proceeds clockwise. The player clockwise of the dealer takes the first turn.
* **Dealer Rotation:** The first dealer is chosen at random. The deal passes clockwise at the start of each new Game Round.

## 3. Melds & Game Round Objectives
* **A "3" (Set):** At least 3 cards of the same rank, regardless of suit. A set may contain more than 3 cards, and duplicate cards (same rank and suit) are allowed.
* **A "4" (Run):** At least 4 consecutive cards of the same suit.
    * An Ace can be low (`A-2-3-4`) or high (`J-Q-K-A`). A single run may use an Ace at both ends (`A-2-…-K-A`, at most 14 cards).
    * **No wrap-around:** A run cannot cross the King-Ace-Two boundary in the middle (e.g. `K-A-2-3` is invalid). An Ace can only be an end card.

Separate melds are independent, and the double deck makes all of the following legal:
* Two sets may share a rank (two sets of 7s).
* Two runs may share a suit. They may be adjacent (`♠2-3-4-5` and `♠6-7-8-9`), partly overlap (`♠3-4-5-6` and `♠4-5-6-7`), or be identical (`♠3-4-5-6` twice).

**The 7 Game Round Objectives:**
1. Two 3s `[3, 3]`
2. One 3, One 4 `[3, 4]`
3. Two 4s `[4, 4]`
4. Three 3s `[3, 3, 3]`
5. Two 3s, One 4 `[3, 3, 4]`
6. One 3, Two 4s `[3, 4, 4]`
7. Three 4s `[4, 4, 4]`

Objectives are exact: a player goes down with exactly the required number of sets and runs, no more and no fewer. For example, a player cannot lay three 3s when the objective is two 3s. Any other cards may only reach the table by being placed on existing melds (Section 6).

## 4. Standard Turn Flow
Unless interrupted by a May-I, a turn consists of:
1. **Pickup:** Take the top card of either the Stock Pile or the Discard Pile. A player may take the top discard and discard that same card later in the turn. Only the most recent discard can ever be taken: once a card is covered by another discard it stays buried, even if the card above it is later taken (by the active player or by a May-I). The active player's discard covers it again at the end of the turn.
2. **Action (Optional):** Go down (if the objective is met) and/or, if already down (including earlier this turn), place Key Cards onto table melds.
3. **Discard:** Place one card face-up on the Discard Pile. If the player has no cards left after the Action step, they have gone out and do not discard.

## 5. The "May-I" Mechanic
A May-I is the only way to take a discard when it is not your turn. Each player may call at most **3 May-Is per Game Round**.

**Resolution Protocol (Priority Queue):**
1. **Trigger:** A May-I opportunity opens only when the active player declines the top discard and chooses to draw from the Stock Pile. If the active player takes the discard, the opportunity is gone. Choosing the Stock Pile is final: once a May-I is called, the active player cannot switch to taking the discard.
2. **Eligibility:** The following players cannot May-I:
    * the active player;
    * the player who discarded the targeted card (there is none for the opening upcard);
    * any player who has already gone down;
    * any player who has used all 3 May-Is this Game Round.
3. **Queue:** The offer passes clockwise, starting with the player immediately clockwise of the active player, and skips ineligible players.
4. **Resolution:**
    * If a player calls May-I, they take the targeted discard **plus the top card of the Stock Pile**, so their hand grows by 2. The May-I immediately ends.
    * If a player passes, the offer moves to the next eligible player.
5. **Resuming Play:** Whether or not someone called May-I, the active player then draws from the Stock Pile and continues their turn normally.

The opening upcard can be May-I'd like any other discard.

## 6. Going Down, Placing Key Cards & Going Out
* **Going Down:** Once a player's hand contains the exact objective (Section 3), they may lay those melds face-up during the Action step of their turn.
* **Placing Key Cards:** Once down, a player may place Key Cards onto any meld on the table, their own or an opponent's. They may do this on the same turn they go down. If a card fits more than one meld, the player chooses which one it goes on.
    * Sets can be extended with any card of that rank.
    * Runs can only be extended at their ends, keeping the suit and sequence. A run ending in a high Ace (`…-K-A`) cannot be extended at that end, and a run starting with a low Ace (`A-2-…`) cannot be extended at that end. For example, `J-Q-K-A` can only take the 10, and `A-2-3-4` can only take the 5.
* **Permanence:** Melds are never rearranged, split or merged once placed. Placement is permanent for the rest of the Game Round.
* **Going Out:** The Game Round ends immediately when a player has no cards left in their hand, whether their last card was discarded or placed on a meld. A final discard is not required. A player can go down and out on the same turn.

## 7. Stock Exhaustion
As soon as the Stock Pile becomes empty, all of the Discard Pile except its top card is shuffled to form a new Stock Pile. The top card stays face-up as the Discard Pile. The rebuild happens straight after whichever draw emptied the stock, so the stock is never empty when someone needs to draw from it. For example, if the stock holds one card when a May-I is called, the May-I caller takes that card. The stock is then rebuilt, and the active player draws from the new stock.

## 8. Stalemate
If every player is down and no card could be placed on any meld on the table, the Game Round ends immediately in a stalemate. This happens when every set holds all 8 cards of its rank, and every run end is either an Ace or both copies of the next card are already on the table.

Every player scores **0 points** for a stalemated Game Round, whatever is left in their hand.

## 9. Scoring
Except after a stalemate (Section 8), the player who goes out scores **0 points** and every other player scores penalty points for **each card still in their hand**. Cards already on the table do not count.

**Holding melds does not help:** Cards that form valid melds in a player's hand still count as penalty points if that player has not gone down and laid them on the table before the Game Round ends.

* **Ranks 2 through 7:** 5 points each
* **Ranks 8 through King:** 10 points each
* **Aces:** 20 points each

After Game Round 7, the player with the lowest cumulative score wins. Players tied for the lowest score share a draw.
