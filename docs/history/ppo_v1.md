# PPO V1 historical baseline

PPO V1 was the project's first learning baseline. It established masked-action
experiments, canonical player observations, and alternating self-play against
an explicitly frozen opponent. That lineage reached the Red14/Blue13 stable
pair; Blue14 was not promoted because evaluation showed forgetting against an
older opponent.

Human play and later audits exposed important limitations. The policies were
strongly capture-oriented, defended inconsistently, and did not exhibit a
reliable territory objective. More importantly, the V1 engine did not match
the physical game's rules: it allowed pure suicide as a selectable immediate
loss, blocked all established territory rather than only opponent territory,
had no PASS action or castle inventory, and used automatic no-placement
termination.

The active project therefore moved to the audited Rules V2 engine and a shared
AlphaZero policy-value network with MCTS. V1 behavior must not be treated as
the current rules or used interchangeably with 82-action V2/V3 checkpoints.

The exact V1 code, tests, reports, and resume documentation remain recoverable
from the annotated Git tag `ppo-v1-final`.
