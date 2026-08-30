# Great Kingdom Rules V2 Audit

This audit separates the physical game's rules from the repository's historical
V1 RL semantics. V1 checkpoints remain unchanged and are not compatible with the
82-action V2 engine.

## Evidence used

- **Publisher documentation:** Korea Boardgames, “이세돌의 위즈스톤 시리즈”
  (2023-05-31):
  <https://www.koreaboardgames.com/magazine/menuDetail?boardCd=contents&postNo=179>
- **User-confirmed rules:** the rule clarifications supplied for this Rules V2
  implementation on 2026-08-30.
- **Secondary material reviewed but not treated as official:** community summaries
  and reviews, including a claim that an empty region using all four board edges is
  not territory:
  <https://namu.moe/w/%EA%B7%B8%EB%A0%88%EC%9D%B4%ED%8A%B8%20%ED%82%B9%EB%8D%A4>.
  No publisher rule text confirming that special exception was found.

## [OFFICIAL / DOCUMENTED]

- The board is 9×9 (81 placement points).
- One neutral castle starts at the center.
- Blue is the first player; the repository calls the second player's orange castles
  “Red” for continuity.
- Each player has 40 castles.
- On a turn a player either places one castle or passes.
- A player immediately wins after surrounding/capturing any opposing castle or group.
- Two consecutive passes end the game and trigger territory scoring.
- If there is no capture, territory count decides the winner.
- Player castles, the board edge, and the neutral castle can bound territory.
- An empty region containing/touching an opposing castle is not completed territory.
- **Scoring threshold:** Blue wins only when Blue territory is at least two greater
  than Red territory. Otherwise Red wins. There is no draw branch in this rule.
- No Go-style ko machinery is used. Capture ends the game immediately, and V2 has no
  previous-board, ko-point, superko, or repetition-history state.

## [USER-CONFIRMED]

### Opponent territory

A player may not place inside the opponent's currently completed territory. The move
is illegal, with no board, turn, pass-count, or inventory change.

### Own territory

A player may place inside their own completed territory. The occupied point then stops
counting as empty territory, so such a move can reduce the player's territory score.
Own territory must never be masked merely because it is territory.

### Pure suicide

A placement that captures no opponent and leaves the newly placed castle's connected
group with zero liberties is illegal. The candidate is rolled back, the turn stays with
the player, and inventory is unchanged. This corrects V1's selectable-suicide loss.

### Capture priority

Capture is checked before pure suicide. If a candidate removes the last liberty from
an opposing group, it is legal and wins immediately even when the placing group would
otherwise have zero liberties.

### Ko

No ko, superko, ko-point, board-history, or repetition prohibition is added.

### Opening territory safeguard

Before both colors have at least one castle on the board, the large connected empty
board is not treated as one player's territory merely because it touches that player's
first castle.

## [UNRESOLVED]

- **All-four-board-edges exception:** a secondary source claims that a region bounded
  using all four board edges is not territory. The publisher documentation confirms
  that board edges can bound territory but does not state this special exception. V2
  therefore adds no all-four-edges rule.
- **Other special territory edge cases:** no additional publisher-confirmed exceptions
  were found. None are guessed or implemented.
- **Exact printed-rulebook phrasing:** the publisher documentation is sufficient to
  resolve the scoring threshold, but a directly accessible official rulebook PDF was
  not available during this audit.

## Scoring decision

The publisher explicitly says that the first player wins with **two or more** extra
territory points; otherwise the second player wins. V2 implements:

```text
Blue wins if blue_territory >= red_territory + 2
otherwise Red wins
```

V2 has no `komi` field and no draw result.

## V1 / V2 rule matrix

| Rule | Status | Evidence | V1 behavior | V2 behavior |
|---|---|---|---|---|
| Placement action | Documented | 9×9 board | 0–80 | 0–80 |
| PASS | Documented | Publisher documentation | Missing | Action 81, always legal while active |
| 2 consecutive passes | Documented | Publisher documentation | Missing; auto-score on no placement | Immediate territory scoring |
| Opponent territory | User-confirmed; publisher-supported | Publisher says opponent territory cannot be entered | All territory blocked | Illegal only for opponent |
| Own territory | User-confirmed | User clarification | Incorrectly blocked | Legal if not otherwise illegal |
| Pure suicide | User-confirmed | User clarification | Selectable, immediate loss | Illegal, rollback, masked |
| Capture priority | User-confirmed; documented capture win | Publisher + user clarification | Capture checked first | Capture checked first and wins |
| Ko | Documented/user-confirmed | Capture-terminal design + clarification | No ko state | No ko state/history |
| Inventory | Documented | 40 castles per player | Not tracked | 40 each; legal placement decrements |
| Scoring | Documented | Publisher: Blue needs a lead of at least 2 | Red territory + 3.0; possible draw | Blue ≥ Red + 2, else Red; no draw |
| No placement available | Documented PASS flow | Publisher documentation | Automatic score end | Game continues; PASS remains available |
| Four-edge exception | Unresolved | Secondary source only | No explicit exception | No explicit exception |
