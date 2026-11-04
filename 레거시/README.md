# HDBSCAN 기반 레짐 인식 및 적응형 PI 제어를 통한 불확실한 광고 경매 환경의 강건한 예산 페이싱

> **2026-2 YBIGTA Conference Project**

## Overview

온라인 광고에서는 사용자가 웹페이지나 앱을 열 때마다 어떤 광고를 보여줄지 결정하는 실시간 경매(Ad Auction)가 이루어진다. 광고주는 정해진 예산을 정해진 기간 동안 적절한 속도로 소진하도록 입찰가 등을 조절해야 한다. 이를 **광고 예산 페이싱(Budget Pacing)** 이라고 한다.

그러나 실제 광고 경매 환경은 시간에 따라 트래픽, 경쟁 강도, 낙찰 가격이 계속 변하는 비정상 환경이다. 특정 상황에서 잘 작동하던 고정 PI 제어기는 다른 상황에서 예산을 너무 빠르게 소진하거나, 반대로 충분히 집행하지 못할 수 있다.

본 프로젝트는 현재 경매 환경의 종합적 운영 상황인 **레짐(Regime)** 을 HDBSCAN으로 식별하고, 레짐에 맞는 PI 계수와 안전 전환 규칙을 적용하는 적응형 예산 페이싱 제어기를 구현한다.

## Problem Definition

고정 PI 제어기는 제어 신호와 실제 지출의 관계가 크게 변하지 않는다고 가정한다. 하지만 실제 광고 경매에서는 같은 입찰 강도라도 트래픽과 경쟁 상황에 따라 실제 지출액이 달라진다.

이때 제어기와 환경 사이에 **파라미터 불일치(Parameter Mismatch)** 가 발생한다. 즉, 고정된 PI 계수는 특정 상황에서는 적합하더라도 다른 상황에서는 적합하지 않다.

이를 해결하기 위해 다음 과정을 구현한다.

1. 광고 경매 상태 로그에서 현재 레짐을 식별한다.
2. 레짐별로 적합한 PI 계수 \(K_p, K_i\)를 탐색한다.
3. 레짐 전환 시 히스테리시스, 출력 제한, 변화율 제한을 적용한다.
4. 고정 PI와 레짐 인지형 PI의 추종 성능과 안정성을 비교한다.

## Input / Output

| 구분           | 내용                                                          |
| ------------ | ----------------------------------------------------------- |
| Input        | 시간대별 트래픽량, 경쟁 강도, 누적 지출액, 남은 예산, 목표 대비 지출 오차, 직전 입찰 승수      |
| Intermediate | HDBSCAN 기반 레짐 판정, 레짐별 PI 계수, 안전 전환 규칙                       |
| Output       | 시간대별 입찰 승수, 실제 지출 경로, 목표 지출 곡선 대비 Tracking Error, 제어 안정성 지표 |

## Research Hypotheses

### H1. Regime Effect

서로 다른 광고 경매 레짐에서는 목표 추종에 적합한 PI 계수 조합이 다르다.

* 레짐별 최적 \(K_p, K_i\) 탐색
* 레짐 × PI 계수 상호작용 검정

### H2. Tracking Effect

레짐 인지형 적응 PI 제어는 전체 환경에서 탐색한 최적 Fixed PI 제어보다 목표 지출 곡선을 더 정확하게 추종한다.

* Tracking Error(TE) 비교
* 예산 미소진·초과 집행 비교

### H3. Stability Effect

히스테리시스, 출력 제한, 변화율 제한은 레짐 전환에 따른 입찰 승수 변동성을 줄인다.

* 레짐 전환 횟수 비교
* 입찰 승수 변동성 및 추종 성능 비교
* 안전 규칙 파라미터 탐색

### H4. Noise Robustness

어느 레짐에도 속하지 않는 HDBSCAN 노이즈와 학습에 없던 이상 상황에서 안전한 대응 정책을 설계한다.

* 트래픽 이상·경쟁 강도 충격 주입
* 노이즈 코호트별 성능 분석
* 최근 레짐 유지 또는 보수적 PI 계수 적용 등 폴백 정책 비교

## Methodology

```mermaid
flowchart LR
    A["광고 경매 시뮬레이션"] --> B["상태 로그 구축"]
    B --> C["HDBSCAN 레짐 인식"]
    C --> D["레짐별 PI 계수 탐색"]
    D --> E["안전 전환 규칙 적용"]
    E --> F["Fixed PI와 성능 비교"]
    F --> G["XAI·통계 검증"]
```

* **Simulation:** Yahoo Budget Pacing Simulation Framework 기반 광고 경매 시나리오 생성
* **Regime Discovery:** HDBSCAN으로 트래픽·경쟁·지출 상태 군집화
* **Control:** Fixed PI, Regime-aware PI, Hysteresis, Output Bound, Rate Limit
* **Optimization:** Optuna 기반 베이지안 최적화
* **Evaluation:** Tracking Error, 예산 소진율, 제어 출력 변동성
* **Statistical Validation:** 혼합효과모형, 부트스트랩 신뢰구간, 상호작용 검정
* **Explainability:** 레짐별 대표 상태, 대리 설명 모델, SHAP 기반 레짐 판정 및 제어 전환 근거 분석

## Repository Structure

```text
.
├── docs/           # 연구 설계·결과 보고서
├── configs/        # 실험 설정값
├── data/           # 시뮬레이션 로그
├── src/            # 재사용 가능한 핵심 구현
│   ├── simulator/
│   ├── controller/
│   ├── regime/
│   ├── optimization/
│   ├── evaluation/
│   └── xai/
├── experiments/    # H1~H4별 실험 실행 코드
├── outputs/        # H1~H4별 그래프·표·결과
└── README.md
```

## Tech Stack

`Python` · `Yahoo Budget Pacing Simulation Framework` · `pandas` · `NumPy` · `scikit-learn` · `HDBSCAN` · `Optuna` · `SHAP` · `SciPy` · `statsmodels` · `MLflow` · `Plotly` · `Docker`

## Team






## Reference

* [Yahoo Budget Pacing Simulation Framework](https://github.com/yahoo/BudgetPacingSimulation)
