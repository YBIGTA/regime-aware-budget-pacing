# CMA-ES 세대 수 비교 · E2 합산

기존 `pacing_gain_mlp_v2`의 B 선형회귀 지출 예측기와 E2 합산 설정을 그대로 두고,
seed 1·2·3을 각각 240세대까지 한 번 학습해 60/120/240세대를 비교한다.

- 후보 수: 세대당 12
- 학습일: 400000~400079 (80일)
- 후보 선택: 400080~400119 (40일)
- 검증: 100000~100199 (200일)
- 최종 테스트 500일: 사용하지 않음

맥미니 실행 작업공간:

```text
/Users/chanwoong/pacing_two_state_20261114
```

진행 상황:

```bash
screen -r cmaes240_20261120
tail -f /Users/chanwoong/pacing_two_state_20261114/research_report/pacing_gain_mlp_cmaes_generations_20261120/logs/progress.log
cat /Users/chanwoong/pacing_two_state_20261114/research_report/pacing_gain_mlp_cmaes_generations_20261120/results/status.txt
```

`screen`에서 빠져나올 때는 `Ctrl-A`, `D`를 누른다. 학습은 계속된다. 같은 `run.sh`를
다시 실행하면 seed별 `resume_seed*.pkl`에서 이어서 실행한다.
