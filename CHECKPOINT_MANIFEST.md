# Great Kingdom RL Checkpoint Manifest

검증 기준일: 2026-08-22

브랜치: `feat/masked-ppo-minimal-e2e-20260822`

모든 SHA-256과 `num_timesteps`는 로컬 ZIP을 직접 읽어 다시 확인했다.
10,000 timestep 요청은 8개 환경의 rollout 단위 때문에 실제 counter를
10,240씩 증가시킨다.

| Logical name | File path | SHA-256 | num_timesteps | 역할 |
|---|---|---|---:|---|
| Red0 | `models/MaskablePPO_CNN/masked_ppo_10000.zip` | `e28340c33406a333940df1fe94eee39b9f78494c4b2b2886cd565f27de27c944` | 10,240 | Random Blue로 학습한 최초 Red checkpoint |
| Blue0 | `models/MaskablePPO_CNN/blue_masked_ppo_10000.zip` | `17f990f29d0f2f6ae09c386561cb700210dc267820eed5f828565d7ffd9992ba` | 10,240 | Random Red로 학습한 최초 Blue checkpoint |
| Red1 | `models/MaskablePPO_CNN/red10k_ft_vs_blue10k_plus10k.zip` | `5dcc2d8b016f55ddc4e2c9abd7c7e860a485e8fd105399b0be7e6ef037fe8883` | 20,480 | Red0를 Frozen Blue0에 continuation |
| Blue1 | `models/MaskablePPO_CNN/blue10k_ft_vs_red1_plus10k.zip` | `4b1bdc51904d8b471c576da57df2019e1ba46dd31a069e3db84bd879d18c3e44` | 20,480 | Blue0를 Frozen Red1에 continuation |
| Red2 | `models/MaskablePPO_CNN/red2_ft_vs_blue1_plus10k.zip` | `f99ce74b54a45039a3fb63be8400795ea5820b247b0eed197976eb0ba2fc66ac` | 30,720 | Red1을 Frozen Blue1에 continuation |
| Blue2 | `models/MaskablePPO_CNN/blue2_ft_vs_red2_plus10k.zip` | `7bdd8f45eca903c43f246fe034b3c5ed3e29ced0298d25c83499b5720ca3f48a` | 30,720 | Blue1을 Frozen Red2에 continuation; 다음 Blue parent |
| Red3 | `models/MaskablePPO_CNN/red3_ft_vs_blue2_plus10k.zip` | `29d7e8116baacb3332930669691754036e3075c599724d71c1f7dc8267ed00a9` | 40,960 | Red2를 Frozen Blue2에 continuation; 다음 frozen Red opponent |

## Resume-critical pair

- **NEXT BLUE PARENT = Blue2**

  `models/MaskablePPO_CNN/blue2_ft_vs_red2_plus10k.zip`
- **NEXT FROZEN RED OPPONENT = Red3**

  `models/MaskablePPO_CNN/red3_ft_vs_blue2_plus10k.zip`

GPU 환경으로 이동할 때 위 두 ZIP을 **반드시 함께 복사**하고 SHA-256을
다시 확인해야 한다. 모델 ZIP은 `.gitignore`의 `models/**/*.zip` 규칙으로
제외되므로 Git commit/push에는 포함되지 않는다.
