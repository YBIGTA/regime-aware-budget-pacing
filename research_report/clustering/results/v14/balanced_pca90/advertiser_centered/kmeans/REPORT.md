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
  "largest_cluster_share": 0.8525545634920635,
  "silhouette_assigned": 0.3074326330135606
}
```

## 다른 기간에서의 분리도

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advertiser_centered | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.8313 | 0.3497 | -0.0046 | 0.3016 | 0.7971 | 1.0000 |
| advertiser_centered | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.7847 | 0.2897 | -0.0046 | 0.3385 | 0.7971 | 1.0000 |

## 발견 기간 하나씩 제외한 재현성

| omitted_period | common_inlier_fraction | inlier_jaccard | ARI_common_inliers |
| --- | --- | --- | --- |
| 16 | 1.0000 | 1.0000 | 0.9804 |
| 17 | 1.0000 | 1.0000 | 0.3134 |
| 18 | 1.0000 | 1.0000 | 0.9396 |
| 19 | 1.0000 | 1.0000 | 0.9548 |

## 광고주·시간 정보만으로 군집 예측

| predictor | evaluated_inliers | accuracy | balanced_accuracy |
| --- | --- | --- | --- |
| global_majority | 4032 | 0.8080 | 0.5000 |
| advertiser_only | 4032 | 0.8080 | 0.5000 |
| time_only | 4032 | 0.9494 | 0.8697 |
| advertiser_and_time | 4032 | 0.9494 | 0.8697 |