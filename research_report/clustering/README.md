## 클러스터링을 통한 시장 레짐 탐색

### 연구 질문과 최종 결론

광고 경매의 공급·가격·광고 가치·입찰 반응·최근 변화를 이용해,
시간대 구분 이상의 명확한 시장 레짐을 발견할 수 있는지 탐색했다.

**본 데이터와 피처 표현, 검토한 군집화 설정에서는 시간대별 변화와
구별되는 명확한 시장 레짐을 확인하지 못했다.**

이는 레짐의 통계적 부재를 입증한 결론이 아니다.
다른 피처·거리·알고리즘에서도 레짐이 존재하지 않는다는 뜻도 아니다.

KMeans의 분할에는 물량·가격 차이라는 경제적 해석이 가능했지만,
별개의 자연적인 시장 상태를 발견했다는 확증으로 사용하지 않았다.
시간과 연결된 상태도 제어에 유용할 수 있으나, 이번 분석에서는
군집별 최적 계수 차이나 군집 사용에 따른 정책 개선을 검증하지 않았다.

### 데이터와 분석 범위

- 공개 AuctionNet 데이터의 광고주 48명 전체.
- 발견·전처리 학습 기간: 16~19.
- 내부 기간 이전 확인: 20~21.
- v13·v14 관측: 6기간 × 48광고주 × 42시점 = 12,096행.
- 각 결정 시점에서 직전 6틱까지의 정보만 사용.
- 20~21은 전체 연구에서 사용한 훈련 기간으로, 새 최종 테스트가 아님.
- 24~27은 이번 군집 분석에 사용하지 않음.
- 광고주별 수준 제거는 발견 기간 광고주별 피처 중앙값을 빼는 방식.
- 같은 광고주·시장·인접 시간이 반복되므로 12,096개의 독립 시장이 아님.
- 새로운 광고주에 대한 일반화 실험은 아님.

### 도메인 피처 설계

v12는 기존 시장 피처 11개를 재사용했다.
v13은 원시 Parquet에서 후보 피처 59개를 다시 추출했다.

| 묶음 | 대표 피처 | 의미 |
|---|---|---|
| 공급 | 광고 기회 수 | 집행 가능한 기회 규모 |
| 가치·변동 | 기대 전환 분위수·평균·집중도, sigma | 기회 가치의 분포 |
| 가격 | 최소 낙찰 비용 분위수·평균·집중도 | 시장 비용 분포 |
| 가격·가치 결합 | 비용/(CPA×pValue) 분위수, 가격·가치 상관 | 기준 입찰의 상대 난이도 |
| 입찰 반응 | 승수0.5·1·2·4의 낙찰률·지출·가치 포착률 | 고정가격 재생의 입찰 반응 |
| 동학 | 직전 변화, 구간 차이, 6틱 표준편차·기울기 | 단기 변화·변동성 |

기대 전환은 pValue·pValueSigma를 이용한 clipped-normal 기대값이다.
실제 구매건수·매출과 다르고, sigma는 실측 예측오차가 아니다.

입찰 반응은 과거 전체 기회가격을 관측하는 플랫폼 관점과
고정 가격 재생을 가정한다. 경쟁자 반응을 포함한 인과 효과가 아니다.
6틱 창·피처 묶음은 공학적 선택이며 TICC/catch22의 정확한 재현이 아니다.

### v12 초기 진단

비교 방법들은 두 군집을 보고했다.
기간20~21의 silhouette는 약0.63~0.68이었으나 가장 큰 군집이
전체의 약95~97%를 차지했다.
이를 다수의 균형 잡힌 시장 레짐으로 해석하지 않았다.
해당 수치는 사용자 제공 콘솔에 근거하며, 남아 있는 v12 원본 결과표를
보존 자료에서 함께 확인한다.

### v13 전처리 문제와 감사 결과

로그 변환 → 발견 기간0.5/99.5% 꼬리 제한 → RobustScaler →
중복 제거 → 묶음별 피처 수 가중 → PCA90%를 적용했다.

후보59개 중40개가 남았으나, 사용자 제공 감사 출력에서 다음 쏠림을 확인했다.

| 표현 | 입찰 반응 묶음 분산 비중 | win_0.5 비중 | log_spend_capacity_0.5 비중 | PCA 첫 축 설명 분산 |
|---|---:|---:|---:|---:|
| 절대 수준 | 0.969977 | 0.721338 | 0.248117 | 0.93521507 |
| 광고주 중심화 | 0.980461 | 0.708671 | 0.271238 | 0.94319080 |

RobustScaler는 분산1을 보장하지 않고 IQR로 나눈다.
win_0.5의 scale은 절대 수준0.000621, 중심화0.000503이었다.
묶음 가중 후 표준편차도 각각24.036151·29.566997로 컸다.

작은 IQR과 드문 큰 값이 결합된 피처가 거리를 지배했고,
묶음별 피처 수 가중만으로 이를 해결하지 못했다.

v13 KMeans에서 입찰 반응 묶음 제거 시 원래 분할과의 ARI는
절대 수준-0.01612, 중심화-0.01706이었다.
다른 묶음을 각각 제거했을 때는 모두1.0이었다.
따라서 분할의 입찰 반응 의존성이 강했다.

중심화 HDBSCAN에서는 광고주 정보만으로 군집을 예측한
balanced accuracy가0.99708이었다.
중앙값 제거만으로 광고주 의존이 사라진다고 볼 수 없었다.

v13의 높은 silhouette를 레짐 발견의 확증으로 사용하지 않는다.
다만 v13→v14에서 여러 요소가 함께 바뀌었으므로 분리도 변화 전체를
scaler 하나의 독립적인 효과로 귀속하지 않는다.

### v14 수정 절차와 거리 설계

1. 발견 기간에서 광고주별 수준 제거 여부를 고정.
2. 기존0.5/99.5% 꼬리 제한 유지.
3. 도메인 내부 상수·강한 중복 피처 제거.
4. StandardScaler로 유효 피처별 분산1.
5. 묶음별 남은 피처 수의 제곱근으로 나눔.
6. 발견 기간 묶음별 총분산1을 실제 검사.

거리 제곱은 묶음별 표준화 피처의 평균 제곱 차이를 합한 형태다.
각 도메인에 동일한 사전 가중을 주는 자체 거리 설계이며,
최적 가중이나 정확한 MFA 구현이라는 주장은 하지 않는다.

주 분석은 광고주 중심화·PCA 없음으로 고정했다.
절대 수준·PCA90%는 민감도 비교다.
이전 기간 점수가 가장 높은 표현을 사후에 주 결과로 고르지 않았다.

꼬리 제한이 드문 실제 충격을 숨길 수 있고 StandardScaler도
희귀 사건에 민감하다. 꼬리 감사 기록과 원시표를 함께 확인한다.

### 발견 기간의 실제 분산 균형

| group | variance | variance_share |
| --- | --- | --- |
| dynamics | 1 | 0.16666667 |
| landscape | 1 | 0.16666667 |
| price | 1 | 0.16666667 |
| quality | 1 | 0.16666667 |
| response | 1 | 0.16666667 |
| supply | 1 | 0.16666667 |

여섯 묶음 각각의 총분산1, 비중 약1/6을 확인했다.
중심화 표현의 이전 기간에서는 묶음 비중이 약9.61~21.93%로 달랐다.
이전 기간에 맞춰 재정규화하지 않고 분포 변화로 기록했다.
이 비중은 PCA 적용 전 공간의 값이다.

### 알고리즘·후보·집계

- KMeans: k=2~6, n_init=10, random_state=42.
- HDBSCAN: 최소 군집 크기2%·5%, min_samples10·30의 네 조합.
- 발견8064행에서 최소 군집 크기는162·404행.
- HDBSCAN: EOM, 기본 allow_single_cluster=False.
- 후보 선택은 발견 기간에서만 수행.
- 유효한 HDBSCAN 후보가 없을 때 남긴 설정은 진단용 fallback.
- KMeans의 선택은 k=1과 비교한 정식 군집성 검정이 아님.
- silhouette는 최대1800개 할당 표본을 고정 seed로 추출해 계산.
- HDBSCAN silhouette는 noise 제외 표본의 값으로 coverage와 함께 해석.
- 기간 제외 재현성: 발견4기간 중 하나씩 제외하고 전처리부터 재학습.
  같은 이전 표본의 공통 inlier에서 ARI를 계산하고 coverage도 보고.
- NA는 계산 불가능·해당 없음이며0점과 다름.
- 서로 다른 표현 공간의 silhouette를 직접 순위 매겨 표현을 고르지 않음.

### v14 주 분석 결과

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction | representation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advertiser_centered | kmeans | 20 | 2 | 1 | 0 | 0.83085317 | 0.33310283 | -0.0046012015 | 0.30143409 | 0.79628817 | 1 | balanced_no_pca |
| advertiser_centered | kmeans | 21 | 2 | 1 | 0 | 0.78472222 | 0.27204776 | -0.0045865168 | 0.34123496 | 0.79628817 | 1 | balanced_no_pca |
| advertiser_centered | hdbscan | 20 | 0 | 0 | 1 | 0 | NA | NA | NA | NA | 0 | balanced_no_pca |
| advertiser_centered | hdbscan | 21 | 0 | 0 | 1 | 0 | NA | NA | NA | NA | 0 | balanced_no_pca |

### 발견 기간 HDBSCAN 후보 전체

| key | family | parameter | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hdbscan_0.02_10 | hdbscan | {"fraction": 0.02, "min_samples": 10} | 0 | 0 | 1 | 0 | NA |
| hdbscan_0.02_30 | hdbscan | {"fraction": 0.02, "min_samples": 30} | 0 | 0 | 1 | 0 | NA |
| hdbscan_0.05_10 | hdbscan | {"fraction": 0.05, "min_samples": 10} | 0 | 0 | 1 | 0 | NA |
| hdbscan_0.05_30 | hdbscan | {"fraction": 0.05, "min_samples": 30} | 0 | 0 | 1 | 0 | NA |

네 후보 모두 군집0개·coverage0이었다.
따라서 이 경우는 발견한 군집의 기간 이전 실패 이전에,
발견 기간에서 해당 기준의 밀도 군집을 검출하지 못한 것이다.

이는 모든 관측이 실제 이상치라는 뜻도,
모든 설정·알고리즘에서 군집이 없다는 뜻도 아니다.

### KMeans 분리도와 재현성

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction | representation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advertiser_centered | kmeans | 20 | 2 | 1 | 0 | 0.83085317 | 0.33310283 | -0.0046012015 | 0.30143409 | 0.79628817 | 1 | balanced_no_pca |
| advertiser_centered | kmeans | 21 | 2 | 1 | 0 | 0.78472222 | 0.27204776 | -0.0045865168 | 0.34123496 | 0.79628817 | 1 | balanced_no_pca |

기간20·21의 silhouette는 각각0.3331·0.2720,
기간 제외 평균 ARI는0.7963이었다.
분할 재현성은 확인됐지만 자연적인 저밀도 경계의 증명은 아니다.

### 시간·광고주 의존 분석

| predictor | evaluated_inliers | accuracy | balanced_accuracy |
| --- | --- | --- | --- |
| global_majority | 4032 | 0.8077877 | 0.5 |
| advertiser_only | 4032 | 0.8077877 | 0.5 |
| time_only | 4032 | 0.94915675 | 0.86921687 |
| advertiser_and_time | 4032 | 0.94915675 | 0.86921687 |

시간 구간은 decision_tick//6이다.
발견 기간의 구간별 최빈 군집을 이전 기간에 적용했다.

시간만 사용한 accuracy=0.94915675,
balanced accuracy=0.86921687이다.
광고주만 사용한 balanced accuracy는0.5였고,
광고주+시간은 시간만 사용한 결과와 같았다.

전체 최빈 군집만 예측해도 높은 accuracy가 나올 수 있어
balanced accuracy를 함께 보고했다.
분할이 시간 정보로 상당 부분 예측됨을 시사하지만,
시간이 모든 구조를 완전히 설명한다고 입증하지는 않았다.

### KMeans 분리 기여 상위 피처

| feature | group | between_variance_fraction |
| --- | --- | --- |
| log_opportunities | supply | 0.72447465 |
| log_cost_capacity | landscape | 0.63321934 |
| price_log_q90 | price | 0.50638527 |
| price_log_q10 | price | 0.48279501 |
| log_spend_capacity_4.0 | response | 0.41871289 |
| price_top10_share | price | 0.32371772 |
| log_spend_capacity_2.0 | response | 0.17520879 |
| log_opportunities_window_std | dynamics | 0.14217189 |

군집 간 분산 비중은 군집을 설명하는 서술 지표다.
인과적 중요도나 제어 성능 기여가 아니다.

### 군집별 원시 피처 중앙값

| split | cluster | log_opportunities | log_cost_capacity | price_log_q90 | price_log_q10 | log_spend_capacity_4.0 | price_top10_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| discovery | 0 | 9.4140972 | 2.8228649 | 0.098016057 | 0.072862046 | 2.580677 | 0.12282672 |
| discovery | 1 | 7.8292325 | 1.5545648 | 0.12037293 | 0.078644574 | 1.1830982 | 0.13715087 |
| transfer | 0 | 9.3556521 | 2.879336 | 0.093741901 | 0.068208506 | 2.5635565 | 0.12492378 |
| transfer | 1 | 7.4067107 | 1.6056785 | 0.13201638 | 0.091672435 | 1.2574082 | 0.1305538 |

그룹0은 상대적으로 물량이 많고 가격이 낮으며,
그룹1은 물량이 적고 가격이 높은 상태에 해당한다.
군집 번호는 임의 라벨이다.

시간과 연결된 상태도 제어에 유용할 수 있다.
다만 군집별 최적 계수 차이·정책 개선을 검증한 것은 아니다.

### 전체 민감도 비교의 원시 수치

저장된 CSV를 반올림 없이 삽입한다.

```csv
variant,family,period,clusters,coverage,noise_fraction,largest_cluster_share,silhouette_assigned,advertiser_AMI,time_bin_AMI,mean_LOPO_ARI,mean_LOPO_common_inlier_fraction,representation
market_absolute,kmeans,20,2,1,0,0.61904761904761907,0.20276453338249584,0.020005608418106918,0.15189871192577706,0.49981047311747162,1,balanced_no_pca
market_absolute,kmeans,21,2,1,0,0.56646825396825395,0.19433243424731395,0.0072229263783515612,0.16910509199978785,0.49981047311747162,1,balanced_no_pca
market_absolute,hdbscan,20,0,0,1,0,,,,,0,balanced_no_pca
market_absolute,hdbscan,21,0,0,1,0,,,,,0,balanced_no_pca
advertiser_centered,kmeans,20,2,1,0,0.83085317460317465,0.33310283132087204,-0.0046012015339452553,0.30143408500597624,0.79628816658891444,1,balanced_no_pca
advertiser_centered,kmeans,21,2,1,0,0.78472222222222221,0.2720477611426671,-0.0045865167946054775,0.34123496240970319,0.79628816658891444,1,balanced_no_pca
advertiser_centered,hdbscan,20,0,0,1,0,,,,,0,balanced_no_pca
advertiser_centered,hdbscan,21,0,0,1,0,,,,,0,balanced_no_pca
market_absolute,kmeans,20,2,1,0,0.82986111111111116,0.3556978222470929,-0.0039466373999145748,0.28705073454440433,0.62468070655106622,1,balanced_pca90
market_absolute,kmeans,21,2,1,0,0.78373015873015872,0.29698700776473413,-0.00415484119789795,0.33193256927312564,0.62468070655106622,1,balanced_pca90
market_absolute,hdbscan,20,0,0,1,0,,,,,0,balanced_pca90
market_absolute,hdbscan,21,0,0,1,0,,,,,0,balanced_pca90
advertiser_centered,kmeans,20,2,1,0,0.83134920634920639,0.34969336090673009,-0.0045783272749119171,0.30155583172962819,0.79705574539218516,1,balanced_pca90
advertiser_centered,kmeans,21,2,1,0,0.78472222222222221,0.28968286223036277,-0.0045543472406795339,0.33854438238040679,0.79705574539218516,1,balanced_pca90
advertiser_centered,hdbscan,20,0,0,1,0,,,,,0,balanced_pca90
advertiser_centered,hdbscan,21,0,0,1,0,,,,,0,balanced_pca90
```

### 논문 근거와 적용 범위

1. Adolfsson, Ackerman, Brownstein,
   To Cluster, or Not to Cluster: An Analysis of Clusterability Methods.
   분할 반환과 군집성 존재를 구분하는 근거.
   https://arxiv.org/abs/1808.08317

2. von Luxburg (2010), Clustering Stability: An Overview.
   안정성을 자연 군집의 충분조건으로 보지 않는 해석의 배경.
   https://arxiv.org/abs/1007.1075

3. Beyer et al. (1999), When Is Nearest Neighbor Meaningful?
   특정 조건에서 고차원 거리 대비가 약해질 수 있다는 이론적 배경.
   이번 실패 원인이 거리 집중이라고 실증한 것은 아니다.
   https://research.cs.wisc.edu/techreports/1998/TR1377.pdf

4. HDBSCAN parameter-selection 공식 문서.
   최소 군집 크기·min_samples와 검출 보수성의 관계.
   https://hdbscan.readthedocs.io/en/latest/parameter_selection.html

5. StandardScaler 공식 문서.
   척도 차이·표준화 및 이상치 민감성의 근거.
   https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.StandardScaler.html

6. Lakens (2017), Equivalence Tests: A Practical Primer.
   의미 있는 효과의 부재를 논하려면 효과 한계가 필요하다는 참고.
   평균 차이용 TOST를 군집에 적용한 실험은 수행하지 않았다.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/

7. AuctionNet: 데이터·경매 도메인 배경.
   https://arxiv.org/abs/2412.10798

8. Deep Landscape Forecasting: 가격 분포·낙찰 확률 표현의 배경.
   https://arxiv.org/abs/1905.03028

### 통계적으로 주장하지 않는 것

- 군집·레짐의 부재를 통계적으로 입증했다고 주장하지 않는다.
- Dip test·동등성 검정·새 최종 테스트를 수행했다고 쓰지 않는다.
- p>0.05를 군집 부재의 증거로 쓰지 않는다.
- 종속 관측을 독립 표본으로 취급한 p값·신뢰구간을 제시하지 않는다.
- silhouette·ARI에 통계적으로 유의하다는 표현을 붙이지 않는다.
- 시간 예측만으로 연속 추세와 이산적 시간 연동 상태를 구분했다고
  주장하지 않는다.
- 거리 집중·표본 부족·정보 부족을 확정 실패 원인으로 쓰지 않는다.
- 일부 설정의 실패를 모든 가능한 군집화의 실패로 일반화하지 않는다.

### 최종 해석

**본 데이터와 피처 표현, 검토한 군집화 설정에서는 시간대별 변화와
구별되는 명확한 시장 레짐을 확인하지 못했다.**

초기의 높은 분리도에는 전처리 분산 쏠림이 영향을 주었다.
분산 균형 보정 후 HDBSCAN은 지정한 네 설정에서 군집을 검출하지 못했다.
KMeans 분할에는 물량·가격 차이라는 해석이 가능했지만
시간 정보로 상당 부분 예측됐다.

이는 레짐이 존재하지 않는다는 증명이 아니라,
이번 분석에서 별도 레짐을 확인하지 못했다는 결과다.
더 깊은 원인 규명은 후속 과제로 남기고 현재 탐색을 마무리했다.

v11의 군집 없는 학습 기반 계수 선택은 별도 실험이다.
이번 분석을 v11 성과의 원인이나 환경 피처의 인과 효과에 대한
증명으로 사용하지 않는다.

### 재현 자료

- [원본 파일 목록·SHA256](file_manifest.csv)
- [v14 전체 요약](results/v14/ALL_transfer_summary.csv)
- [발견 기간 HDBSCAN 후보](results/v14/balanced_no_pca/advertiser_centered/discovery_candidates.csv)
- [시간·광고주 예측](results/v14/balanced_no_pca/advertiser_centered/kmeans/nuisance_prediction.csv)
- [군집별 원시 피처](results/v14/balanced_no_pca/advertiser_centered/kmeans/profiles_raw_median.csv)
- [재분석용 원시 피처 압축본](domain_states_raw.csv.gz)
- [v14 소스](source/cluster_v14/balanced_clusters.py)
- [v13 소스](source/cluster_v13/domain_clusters.py)
- [v12 엔진](source/cluster_v12/cluster_diagnostic.py)

원시 피처12,096행과 원본 경매 행은 서로 다르다.
다중GB 경매 Parquet 전체는 이 묶음에 포함하지 않았다.
v13·v14의 후보·규칙·분산·기간 재현성·묶음 제거 결과는
results 하위에 원 경로 구조로 보존했다.

소스는 문서 생성 시점의 파일이다.
실행 당시와의 일치 여부는 원본 PROTOCOL 코드 해시와 비교해야 한다.