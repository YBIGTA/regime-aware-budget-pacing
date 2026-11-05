import os
for k in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
]:
    os.environ.setdefault(k, "2")

import sys
import json
import hashlib
import traceback
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "cluster_v12/cluster_diagnostic.py"

spec = importlib.util.spec_from_file_location(
    "cluster_v12_reuse", ENGINE
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

if not hasattr(m, "save_json"):
    m.save_json = m.js
if not hasattr(m, "save_csv"):
    m.save_csv = m.csv

CODE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT = ROOT / "cluster_v13/outputs" / ("domain_" + CODE[:8])
m.OUT = OUT

RUN = (
    ROOT / "pacing_v11/auctionnet_lab/runs"
    / "pacing_protocol_v11_1412b714"
)

GROUPS = {}
FEATURES = []


def expected(p, sigma):
    s = np.maximum(sigma, 1e-12)
    a = -p / s
    b = (1 - p) / s
    value = (
        p * (norm.cdf(b) - norm.cdf(a))
        + s * (norm.pdf(a) - norm.pdf(b))
        + norm.sf(b)
    )
    return np.where(
        sigma > 0,
        np.clip(value, 0, 1),
        p,
    )


def tick_features(frame, budget, cpa):
    p = frame.pValue.to_numpy(float)
    sigma = frame.pValueSigma.to_numpy(float)
    cost = frame.leastWinningCost.to_numpy(float)

    if (
        not np.isfinite(np.r_[p, sigma, cost]).all()
        or (cost < 0).any()
        or (sigma < 0).any()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("원시 가격·확률·sigma 값 오류")

    value = expected(p, sigma)
    features = {}
    groups = {}

    def put(group, name, number):
        features[name] = float(number)
        groups[name] = group

    put(
        "supply",
        "log_opportunities",
        np.log1p(len(p)),
    )

    # 가치·가격의 위치, 꼬리, 집중도.
    for name, array in [
        ("quality", value),
        ("price", cost),
    ]:
        scale = 1e6 if name == "quality" else 1

        for q in [.1, .5, .9, .99]:
            put(
                name,
                f"{name}_log_q{int(q * 100)}",
                np.log1p(np.quantile(array, q) * scale),
            )

        put(
            name,
            name + "_log_mean",
            np.log1p(array.mean() * scale),
        )
        put(
            name,
            name + "_zero_fraction",
            np.mean(array == 0),
        )

        total = array.sum()
        top = array[array >= np.quantile(array, .9)].sum()
        put(
            name,
            name + "_top10_share",
            top / total if total > 0 else 0,
        )

    # sigma는 예측오차의 실측값이 아닌 데이터상의 변동 척도.
    put(
        "quality",
        "sigma_log_mean",
        np.log1p(sigma.mean() * 1e6),
    )
    put(
        "quality",
        "sigma_log_q90",
        np.log1p(np.quantile(sigma, .9) * 1e6),
    )
    denominator = sigma.mean() + p.mean()
    put(
        "quality",
        "uncertainty_value_ratio",
        sigma.mean() / denominator if denominator > 0 else 0,
    )

    # CPA*p를 기준으로 필요한 입찰 승수의 분포.
    valid = p > 0
    ratio = cost[valid] / (cpa * p[valid])

    for q in [.1, .5, .9, .99]:
        put(
            "landscape",
            f"log_bid_threshold_q{int(q * 100)}",
            np.log1p(np.quantile(ratio, q))
            if len(ratio) else 0,
        )

    put(
        "landscape",
        "zero_pvalue_fraction",
        np.mean(~valid),
    )

    log_p = np.log1p(p * 1e6)
    log_cost = np.log1p(cost)
    correlation = (
        np.corrcoef(log_p, log_cost)[0, 1]
        if log_p.std() > 1e-12 and log_cost.std() > 1e-12
        else 0
    )
    put(
        "landscape",
        "log_price_quality_correlation",
        correlation,
    )
    put(
        "landscape",
        "log_cost_capacity",
        np.log1p(cost.sum() / (budget / 48)),
    )

    # 同じ過去の機会に仮の入札勝数を適用した反応曲線。
    for alpha in [.5, 1., 2., 4.]:
        win = alpha * cpa * p >= cost
        tag = str(alpha)

        put("response", "win_" + tag, win.mean())
        put(
            "response",
            "log_spend_capacity_" + tag,
            np.log1p(cost[win].sum() / (budget / 48)),
        )
        put(
            "response",
            "value_capture_" + tag,
            value[win].sum() / value.sum()
            if value.sum() > 0 else 0,
        )

    for center in [1., 2.]:
        low = cost[
            center * .9 * cpa * p >= cost
        ].sum() / (budget / 48)
        high = cost[
            center * 1.1 * cpa * p >= cost
        ].sum() / (budget / 48)

        put(
            "response",
            "log1p_spend_slope_" + str(center),
            (np.log1p(high) - np.log1p(low))
            / np.log(1.1 / .9),
        )

    return features, groups


def extract(advertisers):
    rows = []
    schema = {}
    inventory = []

    for period in range(16, 22):
        for index, advertiser in enumerate(advertisers):
            files = list(
                (RUN / "data").glob(
                    f"pack-*/p{period}-a{advertiser}.parquet"
                )
            )
            if len(files) != 1:
                raise ValueError(
                    f"원시 Parquet 개수 오류: {period}/{advertiser}"
                )

            path = files[0]
            inventory.append({
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
            })

            frame = pd.read_parquet(
                path,
                columns=[
                    "timeStepIndex",
                    "pValue",
                    "pValueSigma",
                    "leastWinningCost",
                    "budget",
                    "CPAConstraint",
                ],
            )

            if (
                frame.budget.nunique() != 1
                or frame.CPAConstraint.nunique() != 1
            ):
                raise ValueError("에피소드 안에서 예산·CPA 변경")

            budget = float(frame.budget.iloc[0])
            cpa = float(frame.CPAConstraint.iloc[0])

            if budget <= 0 or cpa <= 0:
                raise ValueError("잘못된 예산·CPA")
            if sorted(frame.timeStepIndex.unique()) != list(range(48)):
                raise ValueError("48틱이 완전하지 않은 에피소드")

            base = []
            for _, group in frame.groupby("timeStepIndex", sort=True):
                features, groups = tick_features(group, budget, cpa)
                base.append(features)
                schema.update(groups)
            del frame

            # decision_tick=t일 때 과거 t-6..t-1만 사용.
            for t in range(6, 48):
                row = dict(base[t - 1])

                for name in [
                    "log_opportunities",
                    "quality_log_mean",
                    "price_log_mean",
                    "log_bid_threshold_q50",
                    "log_spend_capacity_1.0",
                ]:
                    history = np.array([
                        point[name] for point in base[t - 6:t]
                    ])

                    dynamics = {
                        "recent_change": history[-1] - history[-2],
                        "short_long_shift": (
                            history[-3:].mean() - history[:3].mean()
                        ),
                        "window_std": history.std(),
                        "window_slope": np.polyfit(
                            np.arange(6), history, 1
                        )[0],
                    }

                    for suffix, number in dynamics.items():
                        column = name + "_" + suffix
                        row[column] = float(number)
                        schema[column] = "dynamics"

                rows.append({
                    "period": period,
                    "advertiser": advertiser,
                    "decision_tick": t,
                    **row,
                })

            if (index + 1) % 12 == 0:
                print(
                    "RAW_FEATURES",
                    period, index + 1, "/", len(advertisers),
                    flush=True,
                )

    result = pd.DataFrame(rows)
    if (
        len(result) != 6 * len(advertisers) * 42
        or not np.isfinite(
            result[list(schema)].to_numpy(float)
        ).all()
    ):
        raise ValueError("피처 행 수·유한값 검사 실패")

    return result, schema, inventory


class DomainPreprocessor:
    def __init__(self, centered, drop=None):
        self.centered = centered
        self.drop = drop

    def fit(self, frame):
        self.names = [
            name for name in FEATURES
            if GROUPS[name] != self.drop
        ]
        values = frame[self.names].to_numpy(float)

        self.centers = (
            frame.groupby("advertiser")[self.names].median()
            if self.centered else None
        )
        if self.centered:
            values = (
                values
                - self.centers.loc[frame.advertiser].to_numpy()
            )

        # 발견 기간에서만 꼬리·척도·상관·PCA 학습.
        self.lo, self.hi = np.quantile(
            values, [.005, .995], axis=0
        )
        clipped = np.clip(values, self.lo, self.hi)
        self.scaler = RobustScaler().fit(clipped)
        scaled = self.scaler.transform(clipped)

        active = np.where(scaled.std(axis=0) > 1e-8)[0]
        correlation = (
            np.corrcoef(scaled[:, active], rowvar=False)
            if len(active) > 1 else np.ones((1, 1))
        )

        keep = []
        for j in range(len(active)):
            if not any(
                abs(correlation[j, k]) > .95 for k in keep
            ):
                keep.append(j)

        self.indices = active[keep]
        if len(self.indices) < 2:
            raise ValueError("중복 제거 후 유효 피처 2개 미만")

        groups = [
            GROUPS[self.names[i]] for i in self.indices
        ]
        self.weights = np.array([
            1 / np.sqrt(groups.count(group))
            for group in groups
        ])

        weighted = scaled[:, self.indices] * self.weights
        self.pca = PCA(
            n_components=.90, svd_solver="full"
        ).fit(weighted)

        if self.pca.n_components_ < 2:
            self.pca = PCA(
                n_components=2, svd_solver="full"
            ).fit(weighted)

        return self

    def before_pca(self, frame):
        values = frame[self.names].to_numpy(float)
        if self.centered:
            values = (
                values
                - self.centers.loc[frame.advertiser].to_numpy()
            )
        scaled = self.scaler.transform(
            np.clip(values, self.lo, self.hi)
        )
        return scaled[:, self.indices] * self.weights

    def transform(self, frame):
        return self.pca.transform(self.before_pca(frame))

    def metadata(self):
        return {
            "centered": self.centered,
            "features": self.names,
            "kept": [
                self.names[i] for i in self.indices
            ],
            "groups": GROUPS,
            "lo": self.lo,
            "hi": self.hi,
            "weights": self.weights,
            "pca_components": self.pca.n_components_,
            "pca_variance": self.pca.explained_variance_ratio_,
            "caution": (
                "발견 기간 0.5/99.5% 꼬리 제한, "
                "묶음별 피처 수 가중, PCA90%. "
                "논문의 정확한 재현이 아닌 자체 설계."
            ),
        }

    def audit(self, frame):
        values = frame[self.names].to_numpy(float)
        if self.centered:
            values = (
                values
                - self.centers.loc[frame.advertiser].to_numpy()
            )
        return pd.DataFrame({
            "feature": self.names,
            "group": [GROUPS[name] for name in self.names],
            "below_discovery_tail": (
                values < self.lo
            ).mean(axis=0),
            "above_discovery_tail": (
                values > self.hi
            ).mean(axis=0),
        })


def fit(family, parameter, values, hdbscan):
    function = (
        m.fit_model if hasattr(m, "fit_model")
        else m.fit_cluster
    )
    return function(family, parameter, values, hdbscan)


def predict(model, family, values, hdbscan):
    output = m.predict(model, family, values, hdbscan)
    return output[0] if isinstance(output, tuple) else output


def extras(frame, hdbscan):
    discovery = frame[
        frame.period.isin([16, 17, 18, 19])
    ].reset_index(drop=True)
    transfer = frame[
        frame.period.isin([20, 21])
    ].reset_index(drop=True)

    for centered, variant in [
        (False, "market_absolute"),
        (True, "advertiser_centered"),
    ]:
        prep = DomainPreprocessor(centered).fit(discovery)

        m.save_json(
            variant + "/feature_preprocessing.json",
            prep.metadata(),
        )
        m.save_csv(
            variant + "/transfer_tail_audit.csv",
            prep.audit(transfer),
        )

        selection = json.loads(
            (
                OUT / variant
                / "LOCKED_DISCOVERY_SELECTION.json"
            ).read_text()
        )

        for family, choice in selection.items():
            parameter = json.loads(
                choice.get("parameter", choice.get("params"))
            )
            model = fit(
                family, parameter,
                prep.transform(discovery), hdbscan,
            )
            reference = predict(
                model, family,
                prep.transform(transfer), hdbscan,
            )

            # 피처 묶음을 하나씩 제거하고 같은 설정으로 재학습.
            rows = []
            for group in sorted(set(GROUPS.values())):
                alternative_prep = DomainPreprocessor(
                    centered, drop=group
                ).fit(discovery)
                alternative = fit(
                    family,
                    parameter,
                    alternative_prep.transform(discovery),
                    hdbscan,
                )
                labels = predict(
                    alternative,
                    family,
                    alternative_prep.transform(transfer),
                    hdbscan,
                )

                common = (reference >= 0) & (labels >= 0)
                ari = None
                if (
                    common.sum() > 1
                    and len(np.unique(reference[common])) >= 2
                    and len(np.unique(labels[common])) >= 2
                ):
                    ari = float(adjusted_rand_score(
                        reference[common], labels[common]
                    ))

                rows.append({
                    "removed_block": group,
                    "common_inlier_fraction": float(common.mean()),
                    "ARI_common_inliers": ari,
                    "alternative_coverage": float(
                        (labels >= 0).mean()
                    ),
                    "alternative_clusters": len(
                        np.unique(labels[labels >= 0])
                    ),
                })

            relative = variant + "/" + family
            m.save_csv(
                relative + "/feature_block_ablation.csv",
                pd.DataFrame(rows),
            )

            # 군집 간 분산 비중: 인과적 중요도가 아닌 서술 지표.
            valid = reference >= 0
            values = prep.before_pca(transfer)
            target = reference[valid]
            importance = []

            if valid.any():
                for j, index in enumerate(prep.indices):
                    column = values[valid, j]
                    total = np.sum(
                        (column - column.mean()) ** 2
                    )
                    between = sum(
                        np.sum(target == cluster)
                        * (
                            column[target == cluster].mean()
                            - column.mean()
                        ) ** 2
                        for cluster in np.unique(target)
                    )
                    importance.append({
                        "feature": prep.names[index],
                        "group": GROUPS[prep.names[index]],
                        "between_variance_fraction": (
                            float(between / total)
                            if total > 0 else 0
                        ),
                    })

            importance_frame = pd.DataFrame(
                importance,
                columns=[
                    "feature", "group",
                    "between_variance_fraction",
                ],
            ).sort_values(
                "between_variance_fraction",
                ascending=False,
            )
            m.save_csv(
                relative + "/separation_contributors.csv",
                importance_frame,
            )


def main():
    if "--status" in sys.argv:
        for name in ["STATUS.json", "REPORT.md"]:
            path = OUT / name
            if path.exists():
                print(path.read_text())
        return

    OUT.mkdir(parents=True, exist_ok=True)

    import fcntl
    lock = (OUT / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ALREADY RUNNING", OUT)
        return

    if (OUT / "DONE.json").exists():
        print("ALREADY COMPLETE", OUT)
        return

    import hdbscan

    advertisers = json.loads(
        (RUN / "advertisers.json").read_text()
    )
    if len(advertisers) != 48:
        raise ValueError("광고주 48명과 다릅니다.")

    protocol = {
        "code_sha256": CODE,
        "diagnostic_engine_sha256": hashlib.sha256(
            ENGINE.read_bytes()
        ).hexdigest(),
        "discovery": [16, 17, 18, 19],
        "internal_transfer": [20, 21],
        "excluded": [22, 23, 24, 25, 26, 27],
        "prior": "v12 이후의 사후 개발 분석",
        "raw": (
            "기존 훈련 Parquet에서 재추출. "
            "과거 6틱만 사용."
        ),
        "representations": [
            "absolute",
            "discovery-advertiser-median centered",
        ],
        "preprocessing": (
            "로그 변환, 발견 기간 0.5/99.5% 꼬리 제한, "
            "상수·상관계수 절댓값>.95 중복 제거, "
            "도메인 묶음별 피처 수 가중, PCA90%"
        ),
        "scope": (
            "군집 수를 성공 목표로 삼지 않음. "
            "정책 성능 향상을 검증하는 실험 아님."
        ),
        "references": [
            "https://arxiv.org/abs/2412.10798",
            "https://arxiv.org/abs/1905.03028",
            "https://arxiv.org/abs/1706.03161",
            "https://link.springer.com/article/10.1007/s10618-019-00647-x",
            "https://researchonline.jcu.edu.au/47065/",
        ],
    }

    m.save_json("PROTOCOL.json", protocol)
    m.status("extract_domain_features")

    try:
        frame, schema, inventory = extract(advertisers)
        GROUPS.update(schema)
        FEATURES.extend(schema)

        m.FEATURES = FEATURES
        m.LOG_FEATURES = []
        m.Preprocessor = DomainPreprocessor
        m.Preprocess = DomainPreprocessor

        m.save_csv("domain_states_raw.csv", frame)
        m.save_json("feature_groups.json", GROUPS)
        m.save_json("raw_input_inventory.json", inventory)

        results = []
        with threadpool_limits(limits=2):
            for centered in [False, True]:
                results.append(
                    m.analyze(frame, centered, hdbscan)
                )
                m.save_csv(
                    "transfer_summary.csv",
                    pd.concat(results, ignore_index=True),
                )

            m.status("domain_block_ablation")
            extras(frame, hdbscan)

        summary = pd.concat(results, ignore_index=True)
        render = m.markdown if hasattr(m, "markdown") else m.table

        text = (
            "# 경매 도메인 피처 군집 진단\n\n"
            + render(summary)
            + f"\n\n관측 {len(frame)}행, "
            + f"최초 후보 피처 {len(FEATURES)}개.\n\n"
            + "공급·가치/불확실성·가격·가격가치결합·"
            + "입찰반응·과거6틱 동학의 6묶음.\n\n"
            + "## 확인할 파일\n\n"
            + "- feature_preprocessing.json: 남은 피처와 PCA 차원\n"
            + "- transfer_tail_audit.csv: 꼬리값·분포 이동\n"
            + "- separation_contributors.csv: 분리 기여의 서술 순위\n"
            + "- feature_block_ablation.csv: 묶음 제거 시 군집 변화\n"
            + "- nuisance_prediction.csv: 광고주·시간 의존\n"
            + "- leave_period_out_stability.csv: 기간 재현성\n"
            + "- profiles_raw_median.csv: 군집별 실제 피처 중앙값\n\n"
            + "## 해석 제한\n\n"
            + "6틱 창은 공학적 사전 선택이며 TICC나 catch22 "
            + "원 알고리즘의 재현이 아니다. "
            + "짧은 창에 catch22 전체를 적용하지 않았다.\n\n"
            + "입찰 반응은 고정 가격과 과거 전체 기회가격을 아는 "
            + "플랫폼 관측을 가정한다. sigma는 실제 예측오차가 아니다.\n\n"
            + "피처 기여 순위는 인과적 중요도가 아니다. "
            + "다수 군집이 자동으로 유용한 제어 레짐을 뜻하지 않는다.\n\n"
            + "꼬리 제한은 극단값 지배를 줄이지만 드문 실제 충격을 "
            + "숨길 수 있어 raw 표와 tail audit를 함께 확인해야 한다.\n\n"
            + "silhouette는 전처리·축소 공간의 지표이며 "
            + "자연 군집이나 통계적 유의성의 증명이 아니다. "
            + "2D 그림만으로 판정하지 않는다.\n\n"
            + "20~21은 내부 기간 이전 확인이며 새 최종 테스트가 아니다. "
            + "24~27은 이번 분석에서 읽지 않았다.\n\n"
            + "## 참고\n\n"
            + "\n".join(protocol["references"])
        )

        (OUT / "REPORT.md").write_text(
            text, encoding="utf-8"
        )
        m.save_json(
            "DONE.json",
            {
                "status": "COMPLETE",
                "rows": len(frame),
                "features": len(FEATURES),
            },
        )
        m.status("all_done", "COMPLETE")
        print(text, flush=True)
        print("\nREPORT:", OUT / "REPORT.md", flush=True)

    except Exception:
        (OUT / "ERROR.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        m.status("exception", "INCOMPLETE")
        raise


if __name__ == "__main__":
    main()
