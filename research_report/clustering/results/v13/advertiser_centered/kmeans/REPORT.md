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
  "largest_cluster_share": 0.9642857142857143,
  "silhouette_assigned": 0.9193495535965986
}
```

## 다른 기간에서의 분리도

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advertiser_centered | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.9831 | 0.9186 | 0.0093 | 0.0082 | 0.8809 | 1.0000 |
| advertiser_centered | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.9683 | 0.9388 | 0.0198 | 0.0094 | 0.8809 | 1.0000 |

## 발견 기간 하나씩 제외한 재현성

| omitted_period | common_inlier_fraction | inlier_jaccard | ARI_common_inliers |
| --- | --- | --- | --- |
| 16 | 1.0000 | 1.0000 | 0.8793 |
| 17 | 1.0000 | 1.0000 | 0.8093 |
| 18 | 1.0000 | 1.0000 | 0.8924 |
| 19 | 1.0000 | 1.0000 | 0.9427 |

## 광고주·시간 정보만으로 군집 예측

| predictor | evaluated_inliers | accuracy | balanced_accuracy |
| --- | --- | --- | --- |
| global_majority | 4032 | 0.9757 | 0.5000 |
| advertiser_only | 4032 | 0.9757 | 0.5000 |
| time_only | 4032 | 0.9757 | 0.5000 |
| advertiser_and_time | 4032 | 0.9757 | 0.5000 |