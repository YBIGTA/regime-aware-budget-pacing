# 분산 균형을 검사한 군집 진단

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction | representation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_absolute | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.6190 | 0.2028 | 0.0200 | 0.1519 | 0.4998 | 1.0000 | balanced_no_pca |
| market_absolute | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.5665 | 0.1943 | 0.0072 | 0.1691 | 0.4998 | 1.0000 | balanced_no_pca |
| market_absolute | hdbscan | 20 | 0 | 0.0000 | 1.0000 | 0.0000 | NA | NA | NA | NA | 0.0000 | balanced_no_pca |
| market_absolute | hdbscan | 21 | 0 | 0.0000 | 1.0000 | 0.0000 | NA | NA | NA | NA | 0.0000 | balanced_no_pca |
| advertiser_centered | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.8309 | 0.3331 | -0.0046 | 0.3014 | 0.7963 | 1.0000 | balanced_no_pca |
| advertiser_centered | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.7847 | 0.2720 | -0.0046 | 0.3412 | 0.7963 | 1.0000 | balanced_no_pca |
| advertiser_centered | hdbscan | 20 | 0 | 0.0000 | 1.0000 | 0.0000 | NA | NA | NA | NA | 0.0000 | balanced_no_pca |
| advertiser_centered | hdbscan | 21 | 0 | 0.0000 | 1.0000 | 0.0000 | NA | NA | NA | NA | 0.0000 | balanced_no_pca |
| market_absolute | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.8299 | 0.3557 | -0.0039 | 0.2871 | 0.6247 | 1.0000 | balanced_pca90 |
| market_absolute | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.7837 | 0.2970 | -0.0042 | 0.3319 | 0.6247 | 1.0000 | balanced_pca90 |
| market_absolute | hdbscan | 20 | 0 | 0.0000 | 1.0000 | 0.0000 | NA | NA | NA | NA | 0.0000 | balanced_pca90 |
| market_absolute | hdbscan | 21 | 0 | 0.0000 | 1.0000 | 0.0000 | NA | NA | NA | NA | 0.0000 | balanced_pca90 |
| advertiser_centered | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.8313 | 0.3497 | -0.0046 | 0.3016 | 0.7971 | 1.0000 | balanced_pca90 |
| advertiser_centered | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.7847 | 0.2897 | -0.0046 | 0.3385 | 0.7971 | 1.0000 | balanced_pca90 |
| advertiser_centered | hdbscan | 20 | 0 | 0.0000 | 1.0000 | 0.0000 | NA | NA | NA | NA | 0.0000 | balanced_pca90 |
| advertiser_centered | hdbscan | 21 | 0 | 0.0000 | 1.0000 | 0.0000 | NA | NA | NA | NA | 0.0000 | balanced_pca90 |

주 분석은 balanced_no_pca/advertiser_centered. 절대 수준과 PCA90은 민감도 비교다.

발견 기간의 각 유효 피처 분산1, 각 묶음 총분산1을 실제 검사했다. 더 많은 군집이 생긴다는 보장은 없다.

묶음 균등 가중은 연구자의 거리 설계이며 최적 가중이나 정확한 MFA 구현이 아니다.

꼬리 제한이 실제 충격을 숨길 수 있으므로 tail audit와 원시표를 함께 확인한다. StandardScaler도 희귀 사건에 민감하다.

발견 기간의 분산 균형이 이전 기간에서도 유지된다는 보장은 없다. 이전 기간에 맞춰 재표준화하지 않는다.

분리도·noise·공동 inlier 비율·기간 재현성·광고주/시간 의존·묶음 제거를 함께 확인한다. 다른 표현 공간의 silhouette만 직접 비교해 가장 높은 결과를 고르지 않는다.

군집 분리는 최적 계수 차이나 정책 개선의 검증이 아니다. 통계적 유의성을 주장하지 않는다.

## 참고

https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.StandardScaler.html
https://scikit-learn.org/1.5/modules/clustering.html
https://hdbscan.readthedocs.io/en/latest/parameter_selection.html