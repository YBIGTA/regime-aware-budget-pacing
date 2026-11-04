# 실험 조건 설정 및 일괄 실행

Yahoo 원본 코드는 수정하지 않고 팀 저장소의 실행기가 외부 JSON 설정을 읽어
Fixed PI 실험을 수행한다.

## 빠른 시작

설정 파일에서 시나리오, 시드, PI 계수를 수정한 뒤 실행한다.

```bash
.venv/bin/python src/simulator/run_experiments.py \
  --config configs/v1/experiment_1_1.json
```

기본 출력 루트는 `outputs/simulation_runs/<experiment_id>/`이다.

- `runs/<run_id>/trace.csv`가 이미 있는 run은 다시 생성하지 않고 `skipped`로
  `manifest.csv`에 기록한다. 시드나 PI 계수를 설정에 추가한 뒤 같은 명령을
  다시 실행하면 새 조합만 실행된다.
- 실행 중에는 `trace.csv.partial`에 쓰고, 검증과 `metadata.json` 저장이 끝난
  뒤 `trace.csv`로 이름을 바꾼다. 중간에 멈춘 run은 다음 실행 때 처음부터 다시
  계산된다.
- 이미 있는 run도 다시 계산하려면 `--overwrite`를 추가한다.

진행률은 터미널에 표시된다. 터미널에서는 한 줄이 갱신되고, 로그 파일로
리다이렉트하면 run마다 10% 단위로 줄이 추가된다.

```
[3/195] baseline_static__seed42__kp50__ki0p5 minute 720/1440 (50.0%) | overall 1.3% elapsed 5m40s ETA 7h03m12s
```

### 병렬 실행

`--workers N`을 주면 run을 프로세스 N개로 나눠 동시에 실행한다. 기본값은 1이며
순차 실행과 같다. run마다 시드로 난수를 초기화하므로 결과는 순차 실행과 같다.
코어 수만큼 주면 된다(맥 M1은 8). 이때 진행률은 run이 끝날 때마다 한 줄씩
출력한다.

```bash
.venv/bin/python src/simulator/run_experiments.py \
  --config configs/v1/experiment_1_1.json --workers 8
```

```
[12/675] baseline_static__seed51__kp1__ki0p01 completed in 2m28s | overall  1.8% elapsed 4m37s ETA 3h55m44s
```

ETA는 남은 시뮬레이션 분 수로 계산한다. 경매가 많은 시나리오는 분당 계산
시간이 길어서 실제 시간과 차이가 날 수 있다.

## 설정 구조

```json
{
  "experiment_id": "experiment_1_1",
  "seeds": [42, 43, 44, 45, 46],
  "pi_coefficients": [
    {"kp": 1.0, "ki": 0.01},
    {"kp": 5.0, "ki": 0.05},
    {"kp": 50.0, "ki": 0.5}
  ],
  "scenarios": [
    {
      "id": "combined_low_to_high_at_noon",
      "minutes": 1440,
      "campaigns": {
        "count": 10,
        "daily_budget": 1000.0
      },
      "initial_condition": {
        "traffic": {
          "dc": 1500,
          "amplitude": 0.2,
          "phase": 0.5
        },
        "competition": {
          "untracked_bid_factor": 0.5,
          "untracked_bid_price_multiplier": 0.5
        }
      },
      "changes": [
        {
          "minute": 720,
          "traffic": {"dc": 4500, "amplitude": 0.9},
          "competition": {
            "untracked_bid_factor": 4.0,
            "untracked_bid_price_multiplier": 2.0
          }
        }
      ]
    }
  ]
}
```

### 실험 전체 항목

| 항목 | 의미 |
| --- | --- |
| `experiment_id` | 출력 폴더에 사용할 실험 ID |
| `seeds` | 각 시나리오를 반복할 환경 난수 시드 목록 |
| `pi_coefficients` | 평가할 Fixed PI의 `(Kp, Ki)` 목록 |
| `scenarios` | 고정 또는 시간변화 환경 목록 |

실행기는 `시나리오 × 시드 × PI 계수`의 모든 조합을 실행한다. 기본은 순차 실행이고,
`--workers`를 주면 여러 개를 동시에 실행한다.

### 시나리오 항목

| 항목 | 의미와 범위 |
| --- | --- |
| `id` | 시나리오 ID. 영문자, 숫자, `.`, `_`, `-` 사용 가능 |
| `minutes` | 실행 길이. 현재 하루 이내인 1~1,440분 지원 |
| `campaigns.count` | tracked campaign 수 |
| `campaigns.daily_budget` | 모든 캠페인에 적용할 일 예산. 생략하거나 `null`이면 Yahoo 분포로 자동 생성 |
| `traffic.dc` | 분당 평균 경매 수의 일평균. 0보다 커야 함 |
| `traffic.amplitude` | 일중 cosine 변동폭. 0~1 |
| `traffic.phase` | 일중 cosine 위상. 0~1 |
| `competition.untracked_bid_factor` | 관련 tracked campaign 수 대비 경쟁 입찰 수 계수 |
| `competition.untracked_bid_price_multiplier` | Yahoo 경쟁 입찰 가격 표본에 곱할 배율 |
| `changes` | 지정한 분부터 적용할 조건 변경 목록 |

`changes`에서는 바뀌는 항목만 적을 수 있다. 생략한 항목은 직전 조건을
유지한다. `minute=720`은 0시를 기준으로 정오부터 새 조건을 적용한다.

## 출력 구조

```text
outputs/simulation_runs/<experiment_id>/
├── conditions.csv
├── manifest.csv
└── runs/
    └── <scenario>__seed<seed>__kp<kp>__ki<ki>/
        ├── metadata.json
        └── trace.csv
```

- `conditions.csv`: 시나리오별 조건과 적용 시작·종료 분을 기록한 조건표
- `manifest.csv`: 모든 조합의 실행·검증 상태, 결과 경로, 행 수, 오류를 기록
- `metadata.json`: 실행에 사용한 시나리오, 시드, PI 계수와 조건표
- `trace.csv`: 캠페인별 분 단위 지출·낙찰·PI 상태와 당시 환경 설정

`manifest.csv`는 각 실행이 끝날 때마다 갱신되므로 장기 배치 도중에도 진행
상태와 실패한 조합을 확인할 수 있다.

### 실제 경쟁 입찰 통계

`trace.csv`에는 설정값인 `untracked_bid_factor` 및
`untracked_bid_price_multiplier`와 별도로, 해당 분에 실제 생성된 경쟁 입찰의
통계가 기록된다. 같은 분의 값은 모든 캠페인 행에 동일하게 들어간다.

| 컬럼 | 의미 |
| --- | --- |
| `competitor_bid_count_total` | 해당 분에 실제 생성된 경쟁 입찰 총수 |
| `competitor_bid_count_mean` | 실제 경매 한 건당 평균 경쟁 입찰 수 |
| `competitor_bid_price_mean` | 실제 경쟁 입찰가 평균 |
| `competitor_bid_price_std` | 실제 경쟁 입찰가의 모집단 표준편차 |
| `competitor_bid_price_max` | 실제 경쟁 입찰가 최댓값 |

경매나 경쟁 입찰이 없는 분의 관련 통계는 `0`으로 기록한다.

### 자동 결과 검증

각 실행 직후 `trace.csv`의 행 수, 캠페인별 분 누락·중복, 조건 변경값,
캠페인 간 환경 통계 일치, 분 지출과 누적 지출의 일치, 남은 예산 계산을
자동으로 검사한다. 별도 파일을 만들지 않고 결과를 기존 출력에 기록한다.

- `manifest.csv`: `validation_status`, `validation_error`
- `metadata.json`: `validation.status`, `validation.error`

검증을 통과하면 실행 상태는 `completed`, 실패하면 `validation_failed`가 된다.

## 재현성

실행마다 다음 상태를 초기화한다.

- Yahoo 전역 `Clock`
- Python 난수 시드
- NumPy 난수 시드
- Yahoo 캠페인 생성 시드
- 경매 수와 사용자 속성 전용 난수 스트림
- untracked bid 가격 전용 난수 스트림

따라서 같은 시나리오와 시드는 반복 실행 시 동일한 CSV를 생성한다. 같은
시나리오·시드에서 `(Kp, Ki)`만 바꿔도 분별 경매 수와 사용자 속성의 난수
스트림은 동일하게 유지된다.

## 단일 실행

기존 `run_fixed_pi.py`도 유지되며 외부 조건 인자를 추가로 지원한다.

```bash
.venv/bin/python src/simulator/run_fixed_pi.py \
  --scenario-id single-check \
  --campaigns 10 \
  --minutes 60 \
  --kp 0.1 \
  --ki 0.001 \
  --seed 42 \
  --daily-budget 100 \
  --dc 1500 \
  --amplitude 0.6 \
  --factor-untracked-bids 2 \
  --untracked-bid-price-multiplier 1 \
  --output outputs/single-check.csv
```

단일 실행도 CSV 옆에 같은 이름의 `.metadata.json` 파일을 생성한다.
