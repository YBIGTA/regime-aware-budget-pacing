# 시간대별 절대 추종 오차

검증 시나리오 200일(`100000~100199`)을 동일한 경매 테이프로 다시 실행하고, 5분 종료 시점마다 아래 값을 계산한다.

```text
100 × |실제 누적 지출 − 목표 누적 지출| / 일일 예산
```

각 비교 그림의 위 패널은 시각별 200일 평균 절대 추종 오차다. 아래 패널은 같은 날짜·같은 시각의 `대조군 − 실험군` 차이에 대한 중앙값과 25~75백분위수다.

## 재실행 검증

- `B_fixed_PI`, `B_state_PI` 상한 1.0은 `../../results/daily_scores.csv`의 일별 지표와 비교한다.
- `B_state_PI` 상한 1.2는 `../unspent_ecdf/cap_1_2_daily.csv`와 비교한다.
- 날짜마다 세 조건의 경매 테이프 해시가 같은지 확인한다.
- 일별 추종 오차와 미소진율 차이가 원본 비율값 기준 `1e-9`를 넘으면 실행을 중단한다.

## 재현

저장소 루트에서 고정 산출물과 같은 `numpy 2.3.5`, `scikit-learn 1.8.0`, `joblib 1.5.3` 환경으로 실행한다.

```bash
python research_report/pacing_abc_state_pi/analysis/tracking_error_profiles/plot_tracking_error_profiles.py
```

## 산출물

- `tracking_error_5min.csv.gz`: 세 조건의 실제 5분 기록 172,800행
- `daily_replay_check.csv`: 재실행 일별 지표와 저장된 값의 대조
- `time_profile_summary.csv`: 그림에 사용한 시간대별 평균·중앙값·사분위수
- `state-pi-tracking-profile.png`, `.svg`: 고정 PI 대 상태별 PI
- `cap-tracking-profile.png`, `.svg`: 상한 1.0 대 1.2
- `summary.json`: 날짜·시각 대응과 재현 검증 결과
