# Results

## Final V5 training

| Metric | Result |
|---|---:|
| Architecture | 12-plane, 128-channel, 6-block residual network |
| Parameters | 1,820,674 |
| Training cycles | 31 |
| BEST promotions / rejected candidates | 20 / 11 |
| Self-play games | 7,936 |
| Generated samples | 219,336 |
| Final replay size | 200,000 |
| Illegal-action violations | 0 |
| Training elapsed | about 14 h 27 min |
| Endings | 7,702 capture / 234 PASS-score |
| PASS actions | 2,296 |
| Overall territory-state fraction | 41.0% |
| Final training-block territory-state fraction | 47.9% |

## Throughput

The optimized runner used 64 concurrent games. Its measured mean inference
batch was 30.11 (maximum 64), with 0.2193 games/s and 6.565 positions/s. This
was approximately 15.82 times the measured V4 sequential throughput. Peak GPU
utilization observed during the engineering run was 71%.

## Value calibration

The final fitted scalar temperature was 1.3098. Across candidate calibrations,
mean validation NLL changed from 0.3773 to 0.3632 and mean Brier score from
0.1124 to 0.1099. In the final training block, the fraction of root values with
`|Q| >= 0.9` was 8.3%, compared with 16.8% in the first block.

## Deterministic internal arena

| Matchup | V5 result | V5 as Blue | V5 as Red |
|---|---:|---:|---:|
| V5 vs V4 iteration 50 | 100-0 | 50/50 | 50/50 |
| V5 vs V3 iteration 50 | 100-0 | 50/50 | 50/50 |
| V5 vs V2 iteration 375 | 100-0 | 50/50 | 50/50 |

These are deterministic internal regression matches over fixed, paired
openings. They should not be interpreted as a human-strength benchmark or as
evidence that the game has been solved.

## Human evaluation and limitations

V5 showed a clear qualitative improvement over V4 in opening, shape, and
strategy. Human-level strategic play was not fully solved. In particular, an
agent already far behind under immediate double-PASS scoring can reduce its own
territory to prolong the game because training optimizes only eventual win or
loss. This is not a PASS implementation bug.

D4 augmentation also did not produce the expected policy consistency: mean
policy L1 disagreement on the fixed symmetry diagnostic rose from 0.2094 to
0.4312, although mean value disagreement fell from 0.2275 to 0.1195. Automated
historical opponents remain an incomplete substitute for strong human review.

Curated machine-readable results are under
[`reports/alphazero_v5_strategy/`](../reports/alphazero_v5_strategy/).
