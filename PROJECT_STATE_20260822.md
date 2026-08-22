# Great Kingdom RL Project State — 2026-08-22

브랜치: `feat/masked-ppo-minimal-e2e-20260822`

## 1. 원래 실패 원인

- occupied/territory invalid action이 action space에 그대로 노출됐다.
- suicide와 실제 impossible action의 semantics가 구분되지 않았다.
- 학습 환경과 실제 플레이에서 invalid action 처리 방식이 달랐다.
- 학습 상대가 고정 heuristic에 머물러 policy 상대 적응을 검증하지 못했다.

## 2. 현재 검증된 semantics

- occupied/territory: action mask로 제외
- suicide: selectable action이며 선택 즉시 패배
- capture: 선택 즉시 승리
- no-playable: 다음 action을 요구하지 않고 즉시 score 종료
- observation: 항상 agent 관점으로 canonicalization
  - channel 0 = agent stones
  - channel 1 = opponent stones
  - channel 2 = blank/neutral

Red와 Blue 양쪽 학습 및 frozen-policy 상대 모두 같은 semantics와 action
mask를 사용한다. 최종 평가의 mask violation은 0이다.

## 3. Random-opponent baseline

| 조건 | 승률 |
|---|---:|
| Random Red vs Random Blue | 약 0.48–0.50 |
| Red PPO 10k vs Random Blue | 약 0.69–0.70 |
| Random Blue vs Random Red | 0.550 |
| Blue PPO 10k vs Random Red | 0.768 |

## 4. Alternating continuation 결과

각 세대는 parent checkpoint를 scratch가 아닌 continuation으로 로드하고,
최신 frozen opponent를 `deterministic=False`로 sampling하면서 10k를 요청했다.

- Red1 adaptation: PASS
- Blue1 adaptation: PASS
- Red2 adaptation: PASS
- Blue2 adaptation: PASS
- Red3 adaptation: PASS
  - Red2 vs Blue2: 0.308
  - Red3 vs Blue2: 0.586
  - delta: +0.278
  - paired bootstrap 95% CI: [+0.226, +0.332]

Red3의 이전 상대 retention도 유지됐다.

- Red2 vs Blue1: 0.554
- Red3 vs Blue1: 0.788
- delta: +0.234
- paired bootstrap 95% CI: [+0.180, +0.286]
- Red3 vs Random Blue: 0.902

현재까지 previous-opponent와 random-opponent 성능은 유지 또는 개선됐고,
명확한 cycling/forgetting 증거는 관찰되지 않았다. 이는 장기 안정성이나
세대 번호에 따른 절대 policy ranking을 증명하지 않는다.

## 5. Generic one-generation runner

`run_alternating_generation.py`는 명시된 learner parent, latest frozen
opponent, previous opponent, output 경로를 사용해 정확히 한 세대만 처리한다.

한 invocation의 범위:

1. 입력·SHA·회귀 검사
2. PRE evaluation
3. parent checkpoint continuation
4. child checkpoint 하나 저장
5. POST, previous-opponent retention, random sanity 평가
6. 동일 seed block의 paired bootstrap CI와 2×2 report 저장
7. STOP

자동 다음 세대, while loop, self-trigger, opponent pool update는 없다.
Red3 실제 실행으로 runner infrastructure와 단일-generation invariant가
검증됐다. 상세 결과는
`reports/red3_finetune_vs_blue2_20260822/`에 있다.

## 6. 다음 GPU resume point

- Blue parent: `models/MaskablePPO_CNN/blue2_ft_vs_red2_plus10k.zip`
- Latest frozen Red opponent: `models/MaskablePPO_CNN/red3_ft_vs_blue2_plus10k.zip`
- Previous Red opponent: `models/MaskablePPO_CNN/red2_ft_vs_blue1_plus10k.zip`

Blue2와 Red3 ZIP은 Git에 포함되지 않으므로 GPU 환경으로 반드시 함께
복사하고 [CHECKPOINT_MANIFEST.md](CHECKPOINT_MANIFEST.md)의 SHA와 대조한다.

다음 명령은 재개용 기록이며 **2026-08-22에는 실행하지 않았다**.

```bash
python run_alternating_generation.py \
  --learner-player blue \
  --parent-model models/MaskablePPO_CNN/blue2_ft_vs_red2_plus10k.zip \
  --latest-opponent models/MaskablePPO_CNN/red3_ft_vs_blue2_plus10k.zip \
  --previous-opponent models/MaskablePPO_CNN/red2_ft_vs_blue1_plus10k.zip \
  --output-model models/MaskablePPO_CNN/blue3_ft_vs_red3_plus10k.zip \
  --report-dir reports/blue3_finetune_vs_red3_<DATE> \
  --timesteps 10000 \
  --episodes 500 \
  --seed <FIXED_SEED>
```

다음 세대도 명시적으로 한 번만 호출한다. 자동 alternating loop는 아직
구현하거나 실행하지 않는다.

## 7. 아직 미확정

- 장기 self-play stability
- 사람 상대 성능
- opponent pool이 필요한 시점
- Nash equilibrium
- 귀 opening의 최적성
- AlphaZero/MCTS 필요 여부
