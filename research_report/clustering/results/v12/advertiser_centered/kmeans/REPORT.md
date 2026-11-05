# advertiser_centered / kmeans

## 발견 기간에서 선택한 설정

```json
{
  "key": "kmeans_2",
  "family": "kmeans",
  "parameter": "{\"k\": 2}",
  "clusters": 2,
  "coverage": 1.0,
  "noise_fraction": 0.0,
  "largest_cluster_share": 0.9690824468085106,
  "silhouette_assigned": 0.6720531161561779
}
```

## 다른 기간에서의 분리도

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advertiser_centered | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.9530 | 0.6813 | 0.0204 | 0.0002 | 1.0000 | 1.0000 |
| advertiser_centered | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.9663 | 0.6446 | 0.0189 | 0.0004 | 1.0000 | 1.0000 |

## 발견 기간 하나씩 제외한 재현성

| omitted_period | common_inlier_fraction | inlier_jaccard | ARI_common_inliers |
| --- | --- | --- | --- |
| 16 | 1.0000 | 1.0000 | 1.0000 |
| 17 | 1.0000 | 1.0000 | 1.0000 |
| 18 | 1.0000 | 1.0000 | 1.0000 |
| 19 | 1.0000 | 1.0000 | 1.0000 |

## 광고주·시간 정보만으로 군집 예측

| predictor | evaluated_inliers | accuracy | balanced_accuracy |
| --- | --- | --- | --- |
| global_majority | 4512 | 0.9597 | 0.5000 |
| advertiser_only | 4512 | 0.9597 | 0.5000 |
| time_only | 4512 | 0.9597 | 0.5000 |
| advertiser_and_time | 4512 | 0.9597 | 0.5000 |