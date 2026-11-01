# 상태 테이블 변수 정의서

이 파일은 `src/preprocessing/state_table.py`의 `COLUMNS`에서 자동으로
생성된다. 직접 고치지 말고 코드를 고친 뒤 다시 생성한다.

```bash
.venv/bin/python src/preprocessing/build_state_table.py --write-docs
```

쓰는 방법과 담당자가 정할 항목은 [state_table.md](state_table.md)에 있다.

## 식별 열

| 열 | 의미 | 단위 | 계산 방법 | 군집화 사용 |
| --- | --- | --- | --- | --- |
| `run_id` | 실행 ID | - | `<scenario>__seed<seed>__kp<kp>__ki<ki>` | 아니오 |
| `scenario_id` | 시나리오 ID | - | `configs/experiment_1_1.json`의 `scenarios[].id` | 아니오 |
| `seed` | 환경 난수 시드 | - | 같은 시나리오·시드면 PI 계수와 무관하게 경매 입력이 같다 | 아니오 |
| `kp` | PI 비례 계수 | - | 실행 설정값 | 아니오 |
| `ki` | PI 적분 계수 | - | 실행 설정값 | 아니오 |
| `campaign_id` | 캠페인 ID | - | campaign_0 ~ | 아니오 |
| `minute` | 하루 중 분 | 분 | 0 ~ 1439 | 아니오 |

## 분할·필터 열

| 열 | 의미 | 단위 | 계산 방법 | 군집화 사용 |
| --- | --- | --- | --- | --- |
| `split` | 데이터 분할 | - | 시드로 결정. 같은 시나리오·시드는 PI와 무관하게 같은 분할 | 아니오 |
| `cluster_row` | 군집화용 대표 행 여부 | bool | 시나리오 x 시드 x 분마다 한 행(가장 작은 kp·ki, 첫 캠페인)만 True | 아니오 (필터용) |

## 구간 채움 표시

| 열 | 의미 | 단위 | 계산 방법 | 군집화 사용 |
| --- | --- | --- | --- | --- |
| `window_filled_15m` | 최근 15분이 모두 찼는지 여부 | bool | minute >= 14이면 True. False인 행의 15m 변수는 있는 분만으로 계산한 값 | 아니오 (필터용) |
| `window_filled_60m` | 최근 60분이 모두 찼는지 여부 | bool | minute >= 59이면 True. False인 행의 60m 변수는 있는 분만으로 계산한 값 | 아니오 (필터용) |

## 환경 변수 (군집화 입력 후보)

| 열 | 의미 | 단위 | 계산 방법 | 군집화 사용 |
| --- | --- | --- | --- | --- |
| `minute_of_day` | 하루 중 시각 | 분 | minute과 같은 값 (0 ~ 1439) | 선택 (시간대로만 군집이 나뉠 수 있음, docs/state_table.md 참고) |
| `env_auctions` | 해당 분 경매 수 | 건/분 | actual_auctions 원값 | 선택 (1분 값, 잡음이 큼) |
| `env_comp_bids_per_auction` | 해당 분 경매 1건당 경쟁 입찰 수 | 건/경매 | competitor_bid_count_mean 원값 | 선택 (1분 값, 잡음이 큼) |
| `env_comp_price_mean` | 해당 분 경쟁 입찰가 평균 | 통화 단위 | competitor_bid_price_mean 원값 | 선택 (1분 값, 잡음이 큼) |
| `env_auctions_mean_15m` | 최근 15분(해당 분 포함, 0분 이전은 없음) 분당 경매 수 평균 | 건/분 | actual_auctions의 구간 평균 | 예 |
| `env_auctions_trend_15m` | 최근 15분(해당 분 포함, 0분 이전은 없음) 경매 수 증감률 | 비율 | 구간 내 actual_auctions의 최소제곱 기울기 x (구간 길이-1) / 구간 평균. 구간 시작부터 끝까지의 상대 변화량. 값이 1개면 0 | 예 |
| `env_auctions_cv_15m` | 최근 15분(해당 분 포함, 0분 이전은 없음) 경매 수 변동성 | 비율 | 구간 actual_auctions의 표준편차(ddof=0) / 구간 평균 | 예 |
| `env_comp_bids_per_auction_15m` | 최근 15분(해당 분 포함, 0분 이전은 없음) 경매 1건당 경쟁 입찰 수 | 건/경매 | 구간 competitor_bid_count_total 합 / 구간 actual_auctions 합 | 예 |
| `env_comp_price_mean_15m` | 최근 15분(해당 분 포함, 0분 이전은 없음) 경쟁 입찰가 평균 | 통화 단위 | 구간 전체 경쟁 입찰의 평균. 분별 평균을 입찰 수로 가중 평균 | 예 |
| `env_comp_price_std_15m` | 최근 15분(해당 분 포함, 0분 이전은 없음) 경쟁 입찰가 표준편차 | 통화 단위 | 분별 평균·표준편차·입찰 수로 계산한 구간 전체의 합동 표준편차(ddof=0) | 예 |
| `env_comp_price_max_15m` | 최근 15분(해당 분 포함, 0분 이전은 없음) 경쟁 입찰가 최댓값 | 통화 단위 | competitor_bid_price_max의 구간 최댓값 | 예 |
| `env_auctions_mean_60m` | 최근 60분(해당 분 포함, 0분 이전은 없음) 분당 경매 수 평균 | 건/분 | actual_auctions의 구간 평균 | 예 |
| `env_auctions_trend_60m` | 최근 60분(해당 분 포함, 0분 이전은 없음) 경매 수 증감률 | 비율 | 구간 내 actual_auctions의 최소제곱 기울기 x (구간 길이-1) / 구간 평균. 구간 시작부터 끝까지의 상대 변화량. 값이 1개면 0 | 예 |
| `env_auctions_cv_60m` | 최근 60분(해당 분 포함, 0분 이전은 없음) 경매 수 변동성 | 비율 | 구간 actual_auctions의 표준편차(ddof=0) / 구간 평균 | 예 |
| `env_comp_bids_per_auction_60m` | 최근 60분(해당 분 포함, 0분 이전은 없음) 경매 1건당 경쟁 입찰 수 | 건/경매 | 구간 competitor_bid_count_total 합 / 구간 actual_auctions 합 | 예 |
| `env_comp_price_mean_60m` | 최근 60분(해당 분 포함, 0분 이전은 없음) 경쟁 입찰가 평균 | 통화 단위 | 구간 전체 경쟁 입찰의 평균. 분별 평균을 입찰 수로 가중 평균 | 예 |
| `env_comp_price_std_60m` | 최근 60분(해당 분 포함, 0분 이전은 없음) 경쟁 입찰가 표준편차 | 통화 단위 | 분별 평균·표준편차·입찰 수로 계산한 구간 전체의 합동 표준편차(ddof=0) | 예 |
| `env_comp_price_max_60m` | 최근 60분(해당 분 포함, 0분 이전은 없음) 경쟁 입찰가 최댓값 | 통화 단위 | competitor_bid_price_max의 구간 최댓값 | 예 |

## 시나리오 설정값 (군집화 금지)

| 열 | 의미 | 단위 | 계산 방법 | 군집화 사용 |
| --- | --- | --- | --- | --- |
| `cfg_traffic_dc` | 설정된 평균 트래픽 규모 | 건/분 | 현재 적용 중인 조건의 traffic.dc | 금지 (시뮬레이터 정답값, 해석·검증용) |
| `cfg_traffic_amplitude` | 설정된 시간대별 변동폭 | 비율 | traffic.amplitude | 금지 (해석·검증용) |
| `cfg_untracked_bid_factor` | 설정된 경쟁 입찰 수 계수 | - | competition.untracked_bid_factor | 금지 (해석·검증용) |
| `cfg_untracked_bid_price_multiplier` | 설정된 경쟁 입찰가 배율 | 배 | competition.untracked_bid_price_multiplier | 금지 (해석·검증용) |
| `cfg_condition_start_minute` | 현재 조건이 시작된 분 | 분 | 전환 시나리오에서 720이면 정오 이후 조건 | 금지 (해석·검증용) |
| `cfg_traffic_mean` | 설정상 기대 경매 수 | 건/분 | 코사인 트래픽 곡선의 해당 분 기댓값 | 금지 (해석·검증용) |

## 제어 결과 (군집화에 쓰지 않음)

| 열 | 의미 | 단위 | 계산 방법 | 군집화 사용 |
| --- | --- | --- | --- | --- |
| `ctrl_daily_budget` | 캠페인 일 예산 | 통화 단위 | daily_budget | 아니오 (제어 결과 분석용) |
| `ctrl_target_spend_cumulative` | 목표 누적 지출 | 통화 단위 | 일 예산 x (minute+1) / 1440 | 아니오 |
| `ctrl_actual_spend_cumulative` | 실제 누적 지출 | 통화 단위 | 해당 분 종료 시점 누적 | 아니오 |
| `ctrl_spend_error` | 지출 오차 | 통화 단위 | 목표 누적 지출 - 실제 누적 지출 (양수면 덜 씀) | 아니오 |
| `ctrl_spend_error_ratio` | 예산 대비 지출 오차 | 비율 | ctrl_spend_error / 일 예산. TE는 이 값의 절댓값을 하루 평균 | 아니오 |
| `ctrl_spend_this_minute` | 해당 분 지출 | 통화 단위 | spend_this_minute | 아니오 |
| `ctrl_wins_this_minute` | 해당 분 낙찰 수 | 건 | wins_this_minute | 아니오 |
| `ctrl_remaining_budget` | 남은 예산 | 통화 단위 | 일 예산 - 실제 누적 지출 | 아니오 |
| `ctrl_pacing_signal_used` | 해당 분에 사용한 입찰 승수 | 배 | 원래 입찰가에 곱한 값 | 아니오 |
| `ctrl_pacing_signal_next` | 다음 분에 사용할 입찰 승수 | 배 | 해당 분 종료 후 PI가 계산한 값 | 아니오 |
| `ctrl_integral_error` | PI 적분 상태 | 비율 x 분 | 누적한 예산 대비 오차 (승수가 상·하한에 걸린 분은 누적하지 않음) | 아니오 |

## 정규화 열

환경 변수마다 `<변수>_z` 열이 있다. 학습 분할의 `cluster_row` 행으로
계산한 평균·표준편차로 정규화한 값이다. 구간 변수의 기준은 그 구간이 다
찬 행(`window_filled_*`가 True)으로만 계산한다. 검증·평가 분할에도 같은
기준을 적용한다. 기준값은 `normalization.json`에 있다.

| 대상 | 군집화 사용 |
| --- | --- |
| `env_*_z` | 원 변수와 같음 |
