# market_absolute / hdbscan

## 발견 기간에서 선택한 설정

```json
{
  "key": "hdbscan_0.02_10",
  "family": "hdbscan",
  "parameter": "{\"fraction\": 0.02, \"min_samples\": 10}",
  "clusters": 0,
  "coverage": 0.0,
  "noise_fraction": 1.0,
  "largest_cluster_share": 0,
  "silhouette_assigned": null
}
```

## 다른 기간에서의 분리도

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_absolute | hdbscan | 20 | 0 | 0.0000 | 1.0000 | 0 | NA | NA | NA | NA | 0.0000 |
| market_absolute | hdbscan | 21 | 0 | 0.0000 | 1.0000 | 0 | NA | NA | NA | NA | 0.0000 |

## 발견 기간 하나씩 제외한 재현성

| omitted_period | common_inlier_fraction | inlier_jaccard | ARI_common_inliers |
| --- | --- | --- | --- |
| 16 | 0.0000 | 0.0000 | NA |
| 17 | 0.0000 | 0.0000 | NA |
| 18 | 0.0000 | 0.0000 | NA |
| 19 | 0.0000 | 0.0000 | NA |

## 광고주·시간 정보만으로 군집 예측

해당 결과 없음