# 최종 미소진율 ECDF

검증일 200일(`100000~100199`)의 실제 일별 최종 미소진율을 사용한 보고서용 그림이다.

- 왼쪽: `B_fixed_PI` 대 `B_state_PI`
- 오른쪽: `B_state_PI` 상한 1.0 대 상한 1.2
- 원본 `unspent` 값은 비율이며 그림에서는 100을 곱해 퍼센트로 표시한다.
- 일별 동률은 미소진율 비율값의 절대 차이가 `1e-12` 이하인 경우로 정의한다.

상한 1.0과 첫 패널의 일별 값은 `../../results/daily_scores.csv`에서 읽는다. 저장되어 있지 않던 상한 1.2의 일별 값은 기존 경매 생성기, B 선형회귀 예측기, 상태별 PI 가중치, anti-windup을 그대로 사용하고 최종 승수 상한만 1.2로 바꿔 다시 실행하며 `cap_1_2_daily.csv`에 저장한다. 모든 날짜의 경매 테이프 해시는 저장된 상한 1.0 결과와 대조한다.

## 재현

저장소 루트에서 고정 산출물과 같은 `numpy 2.3.5`, `scikit-learn 1.8.0`, `joblib 1.5.3` 환경으로 실행한다.

```bash
python research_report/pacing_abc_state_pi/analysis/unspent_ecdf/plot_unspent_ecdf.py
```

## 산출물

- `cap_1_2_daily.csv`: 상한 1.2의 실제 일별 200일 결과
- `summary.json`: 평균, 0.1% 기준 충족 건수, 개선·악화·동률 집계와 입력 경로
- `unspent-ecdf-report.png`, `.svg`: 0.00~0.20% 구간의 보고서용 ECDF와 두 ECDF의 높이 차이
- `unspent-ecdf-appendix.png`, `.svg`: 전체 미소진율 범위의 부록용 ECDF
- `unspent-ecdf.png`, `.svg`: 이전 그림(비교 확인을 위해 보존)
