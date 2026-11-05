# 시장 상태 군집화 진단

발견: 기간16~19. 내부 기간 이전 확인:20~21.
광고주48명, 에피소드별47개 과거 관측.
최종 평가 기간24~27은 읽지 않았다.

## 기간 이전 결과

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_absolute | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.9530 | 0.6709 | 0.0204 | 0.0002 | 0.9026 | 1.0000 |
| market_absolute | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.9663 | 0.6320 | 0.0189 | 0.0004 | 0.9026 | 1.0000 |
| market_absolute | hdbscan | 20 | 2 | 0.9978 | 0.0022 | 0.9530 | 0.6705 | 0.0192 | -0.0001 | 1.0000 | 0.9956 |
| market_absolute | hdbscan | 21 | 2 | 0.9951 | 0.0049 | 0.9628 | 0.6344 | 0.0183 | 0.0001 | 1.0000 | 0.9956 |
| advertiser_centered | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.9530 | 0.6813 | 0.0204 | 0.0002 | 1.0000 | 1.0000 |
| advertiser_centered | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.9663 | 0.6446 | 0.0189 | 0.0004 | 1.0000 | 1.0000 |
| advertiser_centered | hdbscan | 20 | 2 | 0.9982 | 0.0018 | 0.9530 | 0.6785 | 0.0195 | -0.0001 | 1.0000 | 0.9965 |
| advertiser_centered | hdbscan | 21 | 2 | 0.9956 | 0.0044 | 0.9632 | 0.6515 | 0.0183 | 0.0001 | 1.0000 | 0.9965 |

## 지표를 읽는 방법

- clusters: noise(-1)를 제외한 군집 수.
- coverage: 군집에 할당된 비율. 낮으면 일부 점만 묶은 결과다.
- silhouette_assigned: 원래 피처 공간에서 계산한 할당 표본의 분리도. 높을수록 분리되어 있지만 자연 레짐의 증명은 아니다.
- mean_LOPO_ARI: 발견 기간 하나를 빼고 전처리부터 재학습해도 같은 이전 표본을 비슷하게 묶는지 확인한 값. 공통 inlier coverage와 함께 읽는다.
- ARI가 NA면 공통 표본에서 비교할 두 개 이상의 군집이 없는 것이며 성공이 아니다.
- advertiser_AMI/time_bin_AMI: 군집과 광고주·시간대의 연관성. 높으면 시장 상태 대신 광고주/시간대 구분일 가능성을 점검해야 한다.

## 필수 추가 확인

각 방법 폴더의 nuisance_prediction.csv에서 광고주·시간만 사용하는 분류가 군집을 얼마나 재현하는지 확인한다.

market_absolute는 시장 피처의 절대 수준, advertiser_centered는 발견 기간의 광고주별 중앙값을 뺀 피처다. 중심화 후 구조가 사라지면 광고주 고유 수준이 분리를 주도했을 수 있다. 중심화 자체가 정답이라는 뜻은 아니다.

profiles_raw_median.csv로 군집별 가격·물량·전환가치 차이를 확인하고, temporal_transitions.csv와 heatmap으로 같은 광고주 안에서 상태 전환이 있는지 확인한다.

## 해석 제한

KMeans는 연속 분포도 지정 개수로 나눈다. HDBSCAN의 높은 silhouette는 noise 비율과 함께 해석한다. 단일 점수 문턱으로 명확한 레짐을 확정하지 않는다.

군집이 잘 나뉘어도 서로 다른 최적 PI계수가 필요하다거나 정책 성능이 좋아진다는 뜻은 아니다.

13536행은 독립 시장13536개가 아니다. 인접 틱·같은 기간·같은 광고주가 반복된다. 통계적 유의성을 주장하지 않는다.

20~21은 전체 연구에서 이미 사용한 훈련 기간이다. 새로운 최종 테스트가 아니라 이번 군집화의 내부 기간 이전 진단이다.

PCA2차원 그림은 설명용이다. 분리도는 원래 피처 공간에서 계산한다.

## 주의·누락

matplotlib 미설치: 그림 대신 PCA CSV를 확인하세요.

## 참고

https://hdbscan.readthedocs.io/en/latest/parameter_selection.html

https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html

https://scikit-learn.org/1.5/modules/clustering.html