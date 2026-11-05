# market_absolute / hdbscan

## 발견 기간에서 선택한 설정

```json
{
  "key": "hdbscan_0.02_10",
  "family": "hdbscan",
  "parameter": "{\"fraction\": 0.02, \"min_samples\": 10}",
  "clusters": 2,
  "coverage": 0.9973404255319149,
  "noise_fraction": 0.0026595744680850686,
  "largest_cluster_share": 0.9684175531914894,
  "silhouette_assigned": 0.6575143187961001
}
```

## 다른 기간에서의 분리도

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_absolute | hdbscan | 20 | 2 | 0.9978 | 0.0022 | 0.9530 | 0.6705 | 0.0192 | -0.0001 | 1.0000 | 0.9956 |
| market_absolute | hdbscan | 21 | 2 | 0.9951 | 0.0049 | 0.9628 | 0.6344 | 0.0183 | 0.0001 | 1.0000 | 0.9956 |

## 발견 기간 하나씩 제외한 재현성

| omitted_period | common_inlier_fraction | inlier_jaccard | ARI_common_inliers |
| --- | --- | --- | --- |
| 16 | 0.9958 | 0.9993 | 1.0000 |
| 17 | 0.9945 | 0.9980 | 1.0000 |
| 18 | 0.9965 | 0.9989 | 1.0000 |
| 19 | 0.9956 | 0.9991 | 1.0000 |

## 광고주·시간 정보만으로 군집 예측

| predictor | evaluated_inliers | accuracy | balanced_accuracy |
| --- | --- | --- | --- |
| global_majority | 4496 | 0.9613 | 0.5000 |
| advertiser_only | 4496 | 0.9613 | 0.5000 |
| time_only | 4496 | 0.9613 | 0.5000 |
| advertiser_and_time | 4496 | 0.9613 | 0.5000 |