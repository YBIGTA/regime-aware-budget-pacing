# advertiser_centered / hdbscan

## 발견 기간에서 선택한 설정

```json
{
  "key": "hdbscan_0.02_30",
  "family": "hdbscan",
  "parameter": "{\"fraction\": 0.02, \"min_samples\": 30}",
  "clusters": 2,
  "coverage": 0.7139136904761905,
  "noise_fraction": 0.28608630952380953,
  "largest_cluster_share": 0.6898561507936508,
  "silhouette_assigned": 0.761855564226226
}
```

## 다른 기간에서의 분리도

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advertiser_centered | hdbscan | 20 | 2 | 0.7669 | 0.2331 | 0.7326 | 0.7522 | 0.0749 | -0.0013 | 0.1635 | 0.6339 |
| advertiser_centered | hdbscan | 21 | 2 | 0.7440 | 0.2560 | 0.7108 | 0.7406 | 0.0762 | 0.0004 | 0.1635 | 0.6339 |

## 발견 기간 하나씩 제외한 재현성

| omitted_period | common_inlier_fraction | inlier_jaccard | ARI_common_inliers |
| --- | --- | --- | --- |
| 16 | 0.7495 | 0.9521 | -0.0439 |
| 17 | 0.5573 | 0.7317 | 0.1452 |
| 18 | 0.5570 | 0.7083 | 0.2900 |
| 19 | 0.6716 | 0.8787 | 0.2626 |

## 광고주·시간 정보만으로 군집 예측

| predictor | evaluated_inliers | accuracy | balanced_accuracy |
| --- | --- | --- | --- |
| global_majority | 3046 | 0.9554 | 0.5000 |
| advertiser_only | 3046 | 0.9944 | 0.9971 |
| time_only | 3046 | 0.9554 | 0.5000 |
| advertiser_and_time | 3046 | 0.9928 | 0.9787 |