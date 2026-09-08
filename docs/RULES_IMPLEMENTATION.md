# Rules Implementation

`great_kingdom_v2.py` is the sole game-state authority. `gk_env_v2.py`, MCTS,
self-play, evaluation, and both user interfaces delegate legality and state
transitions to it.

## Implemented mechanics

- The board has 9×9 points, a neutral castle at the center, and Blue moves
  first. Blue and Red each begin with 40 castles.
- A turn consists of one legal placement or PASS. Board placements map to
  actions 0–80; PASS is action 81.
- Occupied points and the opponent's current territory are unavailable. Empty
  points in one's own territory remain playable.
- A placement that captures no opposing group and leaves its own connected
  group with no liberty is illegal and leaves all state unchanged.
- Capture is evaluated before suicide. Removing the last liberty of any
  opposing group is legal and ends the game immediately in the placing
  player's favor.
- A placement consumes one castle and resets the consecutive-PASS count. PASS
  consumes no castle. Two consecutive passes end the game by territory score.
- Blue wins a scored game only when Blue territory is at least two points
  greater than Red territory; otherwise Red wins. There is no draw branch.
- No ko, superko, previous-position, or repetition-history rule is implemented.

## Territory model

Territory is derived from the current board on demand. Empty regions are
flood-filled and owned only when they touch castles of exactly one player; the
board edge and neutral castle act as boundaries but do not own a region. Until
both colors have placed at least one castle, the open board is left unclaimed.
Occupied points never count as territory.

The implementation does not add a special exception for a region spanning all
four board edges because that detail was not established by the source material
used for the rules audit. No undocumented special territory cases are guessed.

The public description is independently paraphrased. It does not reproduce an
official rulebook. For publisher context, see Korea Boardgames' official
[introduction to Lee Sedol's games](https://www.koreaboardgames.com/magazine/menuDetail?boardCd=contents&postNo=179).
