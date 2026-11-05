# advertiser_centered / hdbscan

## 발견 기간에서 선택한 설정

```json
{
  "key": "hdbscan_0.02_10",
  "family": "hdbscan",
  "parameter": "{\"fraction\": 0.02, \"min_samples\": 10}",
  "clusters": 2,
  "coverage": 0.9976728723404256,
  "noise_fraction": 0.002327127659574435,
  "largest_cluster_share": 0.9684175531914894,
  "silhouette_assigned": 0.6784165289489392
}
```

## 다른 기간에서의 분리도

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advertiser_centered | hdbscan | 20 | 2 | 0.9982 | 0.0018 | 0.9530 | 0.6785 | 0.0195 | -0.0001 | 1.0000 | 0.9965 |
| advertiser_centered | hdbscan | 21 | 2 | 0.9956 | 0.0044 | 0.9632 | 0.6515 | 0.0183 | 0.0001 | 1.0000 | 0.9965 |

## 발견 기간 하나씩 제외한 재현성

| omitted_period | common_inlier_fraction | inlier_jaccard | ARI_common_inliers |
| --- | --- | --- | --- |
| 16 | 0.9965 | 0.9991 | 1.0000 |
| 17 | 0.9965 | 0.9989 | 1.0000 |
| 18 | 0.9969 | 0.9987 | 1.0000 |
| 19 | 0.9960 | 0.9989 | 1.0000 |

## 광고주·시간 정보만으로 군집 예측

| predictor | evaluated_inliers | accuracy | balanced_accuracy |
| --- | --- | --- | --- |
| global_majority | 4498 | 0.9611 | 0.5000 |
| advertiser_only | 4498 | 0.9611 | 0.5000 |
| time_only | 4498 | 0.9611 | 0.5000 |
| advertiser_and_time | 4498 | 0.9611 | 0.5000 |