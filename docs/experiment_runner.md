# 실험 조건 설정 및 일괄 실행

Yahoo 원본 코드는 수정하지 않고 팀 저장소의 실행기가 외부 JSON 설정을 읽어
Fixed PI 실험을 수행한다.

## 빠른 시작

예제 설정을 복사해 시나리오, 시드, PI 계수를 수정한 뒤 실행한다.

```bash
.venv/bin/python src/simulator/run_experiments.py \
  --config configs/experiment_1_1.example.json
```

기본 출력 루트는 `outputs/simulation_runs/<experiment_id>/`이다. 같은
`experiment_id`의 결과가 있으면 덮어쓰지 않고 실패한다. 의도적으로 같은 실행
ID를 다시 계산할 때만 `--overwrite`를 추가한다.

시뮬레이션 전에 조건표와 실행 조합만 검토하려면 `--plan-only`를 사용한다.

```bash
.venv/bin/python src/simulator/run_experiments.py \
  --config configs/experiment_1_1.representative.json \
  --plan-only
```

이 명령은 경매를 실행하지 않고 `conditions.csv`와 `run_plan.csv`만 생성한다.

## 설정 구조

```json
{
  "experiment_id": "traffic_competition_v1",
  "seeds": [42, 43],
  "pi_coefficients": [
    {"kp": 0.1, "ki": 0.001},
    {"kp": 0.5, "ki": 0.01}
  ],
  "scenarios": [
    {
      "id": "low_to_high_at_noon",
      "minutes": 1440,
      "campaigns": {
        "count": 10,
        "daily_budget": 100.0
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

실행기는 `시나리오 × 시드 × PI 계수`의 모든 조합을 순차 실행한다.

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
- `run_plan.csv`: 실행 예정인 시나리오·시드·PI 계수 조합
- `manifest.csv`: 모든 조합의 성공/실패, 결과 경로, 행 수, 오류를 기록
- `metadata.json`: 실행에 사용한 시나리오, 시드, PI 계수와 조건표
- `trace.csv`: 캠페인별 분 단위 지출·낙찰·PI 상태와 당시 환경 설정

`manifest.csv`는 각 실행이 끝날 때마다 갱신되므로 장기 배치 도중에도 진행
상태와 실패한 조합을 확인할 수 있다.

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
