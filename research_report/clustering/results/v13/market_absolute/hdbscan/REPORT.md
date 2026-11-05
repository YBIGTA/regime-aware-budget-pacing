# market_absolute / hdbscan

## 발견 기간에서 선택한 설정

```json
{
  "key": "hdbscan_0.02_10",
  "family": "hdbscan",
  "parameter": "{\"fraction\": 0.02, \"min_samples\": 10}",
  "clusters": 2,
  "coverage": 0.6257440476190477,
  "noise_fraction": 0.37425595238095233,
  "largest_cluster_share": 0.5963541666666666,
  "silhouette_assigned": 0.4111653875978512
}
```

## 다른 기간에서의 분리도

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_absolute | hdbscan | 20 | 2 | 0.6726 | 0.3274 | 0.6503 | 0.4257 | 0.0090 | 0.0044 | 0.7355 | 0.4953 |
| market_absolute | hdbscan | 21 | 2 | 0.6657 | 0.3343 | 0.6493 | 0.3740 | 0.0029 | 0.0091 | 0.7355 | 0.4953 |

## 발견 기간 하나씩 제외한 재현성

| omitted_period | common_inlier_fraction | inlier_jaccard | ARI_common_inliers |
| --- | --- | --- | --- |
| 16 | 0.0685 | 0.1023 | NA |
| 17 | 0.5747 | 0.8578 | 0.4710 |
| 18 | 0.6691 | 0.7963 | NA |
| 19 | 0.6691 | 0.9224 | 1.0000 |

## 광고주·시간 정보만으로 군집 예측

| predictor | evaluated_inliers | accuracy | balanced_accuracy |
| --- | --- | --- | --- |
| global_majority | 2698 | 0.9711 | 0.5000 |
| advertiser_only | 2698 | 0.9711 | 0.5000 |
| time_only | 2698 | 0.9711 | 0.5000 |
| advertiser_and_time | 2698 | 0.9681 | 0.5047 |