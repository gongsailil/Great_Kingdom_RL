# V4 Human acceptance

자동 평가 이후 **Human Blue 3게임 + Human Red 3게임**, 총 6게임을 직접 완료해 주세요.
최종 V4 ACCEPT/REJECT는 자동 arena 결과와 이 관찰을 함께 보고 결정합니다.
자동 평가만으로 최종 ACCEPT하지 않습니다.

프로젝트 root에서 `.venv`를 활성화한 후 실행합니다.

```bash
python play_vs_alphazero_v4.py \
  --checkpoint runs/alphazero_v4/stability_20260903/latest.pt \
  --human-player blue --mcts-simulations 256

python play_vs_alphazero_v4.py \
  --checkpoint runs/alphazero_v4/stability_20260903/latest.pt \
  --human-player red --mcts-simulations 256
```

클릭 = placement, **P = PASS**, **R = restart**, 창 닫기 = 종료입니다.
Human Red이면 AI Blue가 첫 수를 둡니다. AI는 MCTS256, c_puct1.5,
noise OFF, temperature 0, V4 exact tactical solver ON으로 플레이합니다.
계산 중에는 입력 처리가 잠시 지연될 수 있습니다.

각 게임은 `human_games/v4_20260906/game_NNN.json`에 자동 저장됩니다.
합법 수마다 atomic save하며 restart/창 닫기 시 미완료 게임도 보존합니다.
`status=completed`인 게임 6개를 평가 대상으로 사용하세요.
로그에는 색/역할/좌표/PASS, 착수 전 territory와 pass count, tactical mode,
AI root top-5, 승자와 종료 사유가 포함됩니다. Exact win shortcut은 MCTS를
실행하지 않으므로 visits/prior는 null이며 exact Q=+1로 기록됩니다.
로그는 Git ignored 로컬 파일이며 모델이나 사용자 게임을 자동 업로드하지 않습니다.

각 완료 게임의 파일명과 함께 다음을 짧게 기록해 주세요.

1. 자기 성이 다음 수에 포획될 위기를 실제로 방어하는가?
2. 사람 성을 포획할 기회를 마무리하는가?
3. placement가 이전 V3보다 덜 무작위로 느껴지는가?
4. territory를 만들거나 상대 territory를 방해하는 패턴이 있는가?
5. own territory에 의미 없이 반복 착수하는가?
6. 장기전에서 PASS를 합리적으로 사용하는가?
7. 같은 이상 행동이 반복되는가? 해당 ply 번호를 기록한다.

Capture 종료가 많다는 사실만으로 실패로 보지 않습니다. D/E/F/G strategic
진단도 단일 정답을 강제하지 않습니다. 사람의 관찰 결과는 사용자가 기록하고,
로그를 검토한 뒤 최종 acceptance를 판단합니다.
