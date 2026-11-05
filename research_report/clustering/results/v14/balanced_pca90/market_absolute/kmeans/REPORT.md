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
  "largest_cluster_share": 0.8515625,
  "silhouette_assigned": 0.299582655679234
}
```

## 다른 기간에서의 분리도

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_absolute | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.8299 | 0.3557 | -0.0039 | 0.2871 | 0.6247 | 1.0000 |
| market_absolute | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.7837 | 0.2970 | -0.0042 | 0.3319 | 0.6247 | 1.0000 |

## 발견 기간 하나씩 제외한 재현성

| omitted_period | common_inlier_fraction | inlier_jaccard | ARI_common_inliers |
| --- | --- | --- | --- |
| 16 | 1.0000 | 1.0000 | 0.9908 |
| 17 | 1.0000 | 1.0000 | 0.4368 |
| 18 | 1.0000 | 1.0000 | 0.1220 |
| 19 | 1.0000 | 1.0000 | 0.9492 |

## 광고주·시간 정보만으로 군집 예측

| predictor | evaluated_inliers | accuracy | balanced_accuracy |
| --- | --- | --- | --- |
| global_majority | 4032 | 0.8068 | 0.5000 |
| advertiser_only | 4032 | 0.8068 | 0.5000 |
| time_only | 4032 | 0.9457 | 0.8633 |
| advertiser_and_time | 4032 | 0.9457 | 0.8633 |