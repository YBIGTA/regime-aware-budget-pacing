# STEP 1: Yahoo 시뮬레이터 설치 및 기본 실행

확인일: 2026-11-01

## 원본 확인

- 공식 저장소: <https://github.com/yahoo/BudgetPacingSimulation>
- 확인한 커밋: `2da74bd03253ddad647e21b9a13f00135de27e62`
- 로컬 submodule과 `origin/main`이 같은 커밋임을 `git fetch` 후 확인했다.
- `external/yahoo-budget-pacing/` 아래 파일은 Yahoo 원본이다.
- `src/controller/fixed_pi.py`, `src/simulator/run_fixed_pi.py`,
  `experiments/smoke_test.py`는 이 팀 저장소에서 추가한 파일이며 Yahoo 원본이 아니다.

## 실행 환경

원본의 고정 의존성은 `numpy==1.26.4`, `scipy~=1.11.2`이다. 로컬의 기본
Python 3.14 대신 이 버전들과 호환되는 Python 3.11을 사용했다.

```bash
git submodule update --init --recursive
python3.11 -m venv .venv
.venv/bin/python -m pip install -r external/yahoo-budget-pacing/requirements.txt
```

검증 결과:

- Yahoo 원본 단위 테스트: 20개 모두 통과
- 팀의 1분 smoke test: 정상 종료
- 기본 7일 시뮬레이션: 정상 종료 및 CSV 생성 확인
  - Apple Silicon 단일 코어 기준 실측 시간: 40분 21초
  - Day 0 Hour 0부터 Day 6 Hour 23까지 처리

재현 명령:

```bash
# 팀 저장소 루트에서 실행
.venv/bin/python experiments/smoke_test.py

# Yahoo 원본 테스트
cd external/yahoo-budget-pacing
../../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v

# Yahoo 기본 설정 전체 실행
../../.venv/bin/python -m src.main
```

README는 `src/main.py`를 실행하라고만 안내하지만, 저장소 루트에서
`python src/main.py`를 직접 실행하면 원본의 절대 import 때문에
`ModuleNotFoundError: No module named 'src'`가 발생한다. 원본을 수정하지 않고
`python -m src.main`으로 실행하면 정상 동작한다.

## 핵심 파일과 역할

| 파일 | 역할 |
| --- | --- |
| `src/configuration.py` | 실행 기간, 캠페인 수, 트래픽 파형, 경쟁 입찰 계수와 가격·예산·입찰 분포를 정의한다. |
| `src/main.py` | 시스템을 조립하고 7일간 분 단위 반복을 수행한 뒤 CSV를 쓴다. |
| `src/system_generation_utils.py` | 캠페인의 예산, 입찰 분포, 최대 입찰가, 타기팅 그룹과 페이싱 시스템을 생성한다. |
| `src/system/marketplace.py` | 매분 Poisson 분포로 경매 수를 뽑고 사용자 속성을 생성한 뒤 경매를 실행한다. |
| `src/system/serving_system.py` | 관련 캠페인의 입찰과 untracked 입찰을 생성하고, 낙찰·지출·일별 통계를 갱신한다. |
| `src/system/campaign.py` | 캠페인별 입찰 표본, 예산, 지출, 낙찰 수와 생존 시간 이력을 관리한다. |
| `src/system/auction.py` | 현재 구현된 first-price 경매에서 최고 입찰자를 고르고 입찰가 전액을 결제시킨다. |
| `src/system/clock.py` | 전역 시뮬레이션 시계를 분 단위로 진행한다. |

실행 흐름은 다음과 같다.

```text
main
  -> generate_campaigns / generate_pacing_system
  -> Marketplace.run_iteration (매분)
       -> 경매 수와 사용자 속성 생성
       -> ServingSystem.get_bids
       -> AuctionFP.run
       -> ServingSystem.update_winners / end_iteration
  -> 캠페인별 통계와 전역 통계를 CSV로 저장
```

## 기본 설정에서 자동 생성되는 값

- 캠페인 100개의 일 예산과 총예산
  - 시드 123에서 일 예산 최소 1.524, 중앙값 32.704, 평균 646.138,
    최대 23,226.639
  - 100개 캠페인의 7일 총예산 합계는 452,296.727
- 예산 구간별 tracked bid 가격 분포와 최대 입찰가
- 각 캠페인의 타기팅 그룹
- 매분 경매 수: 평균값이 일중 cosine 파형을 따르는 Poisson 표본
  - `dc=3000`, `amplitude=0.6`, `phase=0.5`
  - 분당 평균값의 범위는 1,200~4,800이고 일평균은 3,000이다.
- 경매별 사용자 속성(`Gender`, `Location`)
- tracked campaign의 입찰 가격
- untracked bid 수와 가격
  - 수는 `round(2 * 관련 tracked campaign 수)`
  - 가격은 `lognorm(s=2.14, scale=exp(-8.19))` 표본
- 최고 입찰자, 결제액, 캠페인별 지출과 낙찰 통계

`campaign_generation_seed_value=123`은 캠페인의 예산과 타기팅 생성에는
적용된다. 그러나 Yahoo 원본 `main.py`는 전역 NumPy 난수 시드를 고정하지
않으므로 경매 수, 사용자 속성, tracked/untracked 입찰 가격까지 완전히
재현하지는 못한다. 전체 실험 재현성은 후속 단계에서 별도로 처리해야 한다.

## 기본 출력

`src/main.py`의 출력 루트는 실행 위치 기준 `../output`이다. 따라서 위 명령을
Yahoo submodule 루트에서 실행하면 다음 위치에 생성된다.

```text
external/output/
  Camp100-UntrackedF2-Traffic3000-0.6-Seed123/
    No_Pacing.csv
```

CSV 하나에 서로 다른 두 표가 연속해서 들어간다.

1. 캠페인별 표: 100행
   - `Campaign`, `Day Started`, `Day Ended`, `Daily Budget`
   - `# Impressions`, `CPM`, `Budget Utilization`, `Overspend`,
     `# Minutes Alive`
   - 일별 또는 시간 구간별 값은 한 CSV 셀 안에 Python list 문자열로 저장된다.
2. 전역 일별 표: 7일 행과 `Overall` 행
   - `Day`, `CPM`, `Spend`, `Overspend`, `# Impressions`,
     `# Over Budget Campaigns`, `# Minutes Alive`

실제 기본 실행 결과는 106KB, 총 110줄이었다. 캠페인 헤더 1줄과 캠페인
100행, 전역 헤더 1줄, 일별 7행과 `Overall` 1행으로 구성된다. 실행한 작업
폴더에서는 다음 파일로 생성되었다.

```text
/Users/chanwoong/Desktop/ybig_conference/output/
  Camp100-UntrackedF2-Traffic3000-0.6-Seed123/
    No_Pacing.csv
```

즉, 원본 출력은 캠페인별·일별 요약 통계이며 HDBSCAN에 필요한 분 단위 경매
환경 로그는 아직 제공하지 않는다. `num_spend_entries_per_day=1`이라 지출은
일 단위이고, 낙찰 수는 `num_win_entries_per_day=24`라 시간 단위로 보관된다.
분 단위 CSV와 실험 메타데이터 분리는 STEP 3에서 추가해야 한다.
