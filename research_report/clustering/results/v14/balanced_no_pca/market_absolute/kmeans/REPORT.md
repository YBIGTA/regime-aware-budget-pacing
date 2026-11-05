# market_absolute / kmeans

## 발견 기간에서 선택한 설정

```json
{
  "key": "kmeans_2",
  "family": "kmeans",
  "parameter": "{\"k\": 2}",
  "clusters": 2,
  "coverage": 1.0,
  "noise_fraction": 0.0,
  "largest_cluster_share": 0.6834077380952381,
  "silhouette_assigned": 0.20371692583284084
}
```

## 다른 기간에서의 분리도

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_absolute | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.6190 | 0.2028 | 0.0200 | 0.1519 | 0.4998 | 1.0000 |
| market_absolute | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.5665 | 0.1943 | 0.0072 | 0.1691 | 0.4998 | 1.0000 |

## 발견 기간 하나씩 제외한 재현성

| omitted_period | common_inlier_fraction | inlier_jaccard | ARI_common_inliers |
| --- | --- | --- | --- |
| 16 | 1.0000 | 1.0000 | 0.3118 |
| 17 | 1.0000 | 1.0000 | 0.7851 |
| 18 | 1.0000 | 1.0000 | 0.6173 |
| 19 | 1.0000 | 1.0000 | 0.2851 |

## 광고주·시간 정보만으로 군집 예측

| predictor | evaluated_inliers | accuracy | balanced_accuracy |
| --- | --- | --- | --- |
| global_majority | 4032 | 0.5928 | 0.5000 |
| advertiser_only | 4032 | 0.5977 | 0.5131 |
| time_only | 4032 | 0.7307 | 0.6703 |
| advertiser_and_time | 4032 | 0.7485 | 0.7059 |