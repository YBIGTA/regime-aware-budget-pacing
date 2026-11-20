# 최근 선형회귀 예측 오차를 이용한 기본 승수 보정

기존 B 선형회귀 지출 예측기와 선택된 E2 합산 계수망을 고정하고, 완료된 과거
구간의 `실제 지출 - 보정 전 예측 지출`을 다음 구간의 기본 승수 선택에 반영한다.

- A: 기존 E2
- B: 공통 EMA 보정 (`eta=0.25`)
- C: 최종 승수 주변 삼각 커널 보정 (`eta=0.25`, `h=0.15`)

맥미니 작업공간:

```text
/Users/chanwoong/pacing_two_state_20261114
```

진행 확인:

```bash
screen -r residual_correction_20261120
tail -f /Users/chanwoong/pacing_two_state_20261114/research_report/pacing_prediction_residual_correction_20261120/logs/progress.log
cat /Users/chanwoong/pacing_two_state_20261114/research_report/pacing_prediction_residual_correction_20261120/results/status.txt
```

예약된 최종 테스트 500일은 사용하지 않는다.
