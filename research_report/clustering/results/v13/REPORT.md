# 경매 도메인 피처 군집 진단

| variant | family | period | clusters | coverage | noise_fraction | largest_cluster_share | silhouette_assigned | advertiser_AMI | time_bin_AMI | mean_LOPO_ARI | mean_LOPO_common_inlier_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_absolute | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.9836 | 0.9191 | 0.0092 | 0.0083 | 0.8938 | 1.0000 |
| market_absolute | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.9702 | 0.9391 | 0.0187 | 0.0079 | 0.8938 | 1.0000 |
| market_absolute | hdbscan | 20 | 2 | 0.6726 | 0.3274 | 0.6503 | 0.4257 | 0.0090 | 0.0044 | 0.7355 | 0.4953 |
| market_absolute | hdbscan | 21 | 2 | 0.6657 | 0.3343 | 0.6493 | 0.3740 | 0.0029 | 0.0091 | 0.7355 | 0.4953 |
| advertiser_centered | kmeans | 20 | 2 | 1.0000 | 0.0000 | 0.9831 | 0.9186 | 0.0093 | 0.0082 | 0.8809 | 1.0000 |
| advertiser_centered | kmeans | 21 | 2 | 1.0000 | 0.0000 | 0.9683 | 0.9388 | 0.0198 | 0.0094 | 0.8809 | 1.0000 |
| advertiser_centered | hdbscan | 20 | 2 | 0.7669 | 0.2331 | 0.7326 | 0.7522 | 0.0749 | -0.0013 | 0.1635 | 0.6339 |
| advertiser_centered | hdbscan | 21 | 2 | 0.7440 | 0.2560 | 0.7108 | 0.7406 | 0.0762 | 0.0004 | 0.1635 | 0.6339 |

관측 12096행, 최초 후보 피처 59개.

공급·가치/불확실성·가격·가격가치결합·입찰반응·과거6틱 동학의 6묶음.

## 확인할 파일

- feature_preprocessing.json: 남은 피처와 PCA 차원
- transfer_tail_audit.csv: 꼬리값·분포 이동
- separation_contributors.csv: 분리 기여의 서술 순위
- feature_block_ablation.csv: 묶음 제거 시 군집 변화
- nuisance_prediction.csv: 광고주·시간 의존
- leave_period_out_stability.csv: 기간 재현성
- profiles_raw_median.csv: 군집별 실제 피처 중앙값

## 해석 제한

6틱 창은 공학적 사전 선택이며 TICC나 catch22 원 알고리즘의 재현이 아니다. 짧은 창에 catch22 전체를 적용하지 않았다.

입찰 반응은 고정 가격과 과거 전체 기회가격을 아는 플랫폼 관측을 가정한다. sigma는 실제 예측오차가 아니다.

피처 기여 순위는 인과적 중요도가 아니다. 다수 군집이 자동으로 유용한 제어 레짐을 뜻하지 않는다.

꼬리 제한은 극단값 지배를 줄이지만 드문 실제 충격을 숨길 수 있어 raw 표와 tail audit를 함께 확인해야 한다.

silhouette는 전처리·축소 공간의 지표이며 자연 군집이나 통계적 유의성의 증명이 아니다. 2D 그림만으로 판정하지 않는다.

20~21은 내부 기간 이전 확인이며 새 최종 테스트가 아니다. 24~27은 이번 분석에서 읽지 않았다.

## 참고

https://arxiv.org/abs/2412.10798
https://arxiv.org/abs/1905.03028
https://arxiv.org/abs/1706.03161
https://link.springer.com/article/10.1007/s10618-019-00647-x
https://researchonline.jcu.edu.au/47065/